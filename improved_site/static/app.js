/* ─────────────────────────────────────────────────────────────────────────
   OralWisdom V3 — app.js
   Handles: image upload → prediction → shade simulation
───────────────────────────────────────────────────────────────────────── */

"use strict";

// ── DOM refs ──────────────────────────────────────────────────────────────

const form            = document.getElementById("uploadForm");
const input           = document.getElementById("imageInput");
const chooseButton    = document.getElementById("chooseButton");
const toast           = document.getElementById("toast");

// Prediction result
const resultArea      = document.getElementById("resultArea");
const resultImage     = document.getElementById("resultImage");
const resultTitle     = document.getElementById("resultTitle");
const jsonLink        = document.getElementById("jsonLink");
const overlayLink     = document.getElementById("overlayLink");
const savePath        = document.getElementById("savePath");
const statTotal       = document.getElementById("statTotal");
const statTooth       = document.getElementById("statTooth");
const statCanine      = document.getElementById("statCanine");
const statIncisor     = document.getElementById("statIncisor");
const runtimeRow      = document.getElementById("runtimeRow");
const runtimeText     = document.getElementById("runtimeText");

// Shade simulation panel
const shadePanel         = document.getElementById("shadePanel");
const vitaGroupTabs      = document.querySelectorAll(".vita-group-tab");
const presetGroups       = document.querySelectorAll(".preset-group");
const presetButtons      = document.querySelectorAll(".preset-btn");
const intensityRow       = document.getElementById("intensityRow");
const intensitySlider    = document.getElementById("intensitySlider");
const intensityValue     = document.getElementById("intensityValue");
const applyCustomBtn     = document.getElementById("applyCustomBtn");
const sliderBrightness   = document.getElementById("sliderBrightness");
const sliderYellow       = document.getElementById("sliderYellow");
const sliderWarmth       = document.getElementById("sliderWarmth");
const outBrightness      = document.getElementById("outBrightness");
const outYellow          = document.getElementById("outYellow");
const outWarmth          = document.getElementById("outWarmth");
const beforeAfterBtn     = document.getElementById("beforeAfterBtn");
const downloadShadeBtn   = document.getElementById("downloadShadeBtn");
const shadeResultWrap    = document.getElementById("shadeResultWrap");
const shadeImage         = document.getElementById("shadeImage");
const shadeLoading       = document.getElementById("shadeLoading");
const shadeBadge         = document.getElementById("shadeBadge");
const shadeSavePath      = document.getElementById("shadeSavePath");

// ── State ─────────────────────────────────────────────────────────────────

let currentImageStem   = null;     // stem returned by /predict
let currentOverlayUrl  = null;     // the segmentation overlay URL
let currentUploadUrl   = null;     // the original uploaded image URL
let currentShadeUrl    = null;     // most recently generated shade URL
let activePreset       = "B1";     // currently selected preset name
let isShowingOriginal  = false;    // before/after toggle state
let shadeDebounceTimer = null;

// ── Toast ─────────────────────────────────────────────────────────────────

let toastTimer = null;

function showToast(message, duration = 0) {
  clearTimeout(toastTimer);
  toast.textContent = message;
  toast.hidden = false;
  toast.classList.remove("toast-error");
  if (duration > 0) {
    toastTimer = setTimeout(() => { toast.hidden = true; }, duration);
  }
}

function showErrorToast(message) {
  clearTimeout(toastTimer);
  toast.textContent = message;
  toast.hidden = false;
  toast.classList.add("toast-error");
  toastTimer = setTimeout(() => { toast.hidden = true; }, 6000);
}

function hideToast() {
  clearTimeout(toastTimer);
  toast.hidden = true;
}

// ── Upload + Predict ──────────────────────────────────────────────────────

function setBusy(isBusy) {
  form.classList.toggle("busy", isBusy);
  chooseButton.disabled = isBusy;
}

async function uploadFile(file) {
  if (!file) return;
  const body = new FormData();
  body.append("image", file);
  setBusy(true);
  showToast("Running V3 model…");

  try {
    const response = await fetch("/predict", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed.");

    hideToast();

    // ── Store state ──────────────────────────────────────────────────
    currentImageStem  = data.image_stem;
    currentOverlayUrl = data.overlay_url;
    currentUploadUrl  = data.upload_url;

    // ── Update prediction result UI ───────────────────────────────────
    resultArea.hidden = false;
    resultImage.src   = `${data.overlay_url}?t=${Date.now()}`;
    resultTitle.textContent = data.image_name;
    jsonLink.href     = data.json_url;
    overlayLink.href  = data.overlay_url;
    savePath.textContent = `Overlay saved: ${data.overlay_path}`;

    const counts = data.counts || {};
    statTotal.textContent   = data.num_instances || 0;
    statTooth.textContent   = counts.tooth    || 0;
    statCanine.textContent  = counts.canine   || 0;
    statIncisor.textContent = counts.incisor  || 0;
    runtimeRow.hidden = false;
    runtimeText.textContent = `Inference: ${data.runtime_seconds}s`;

    // ── Show shade panel & trigger default shade ──────────────────────
    resetShadePanel();
    shadePanel.hidden = false;
    
    restorePanel.hidden = false;
    resetRestorePanel();
    
    shadePanel.scrollIntoView({ behavior: "smooth", block: "start" });

    // Auto-apply the default preset (B1) immediately
    requestShade(activePreset);

  } catch (error) {
    showErrorToast(error.message);
  } finally {
    setBusy(false);
  }
}

// ── Shade Panel ───────────────────────────────────────────────────────────

function resetShadePanel() {
  currentShadeUrl   = null;
  isShowingOriginal = false;
  beforeAfterBtn.setAttribute("aria-pressed", "false");
  beforeAfterBtn.classList.remove("pressed");
  shadeResultWrap.hidden = true;
  downloadShadeBtn.hidden = true;

  // Reset to B1 preset
  selectPreset("B1");
  intensitySlider.value = 100;
  intensityValue.textContent = "100%";
}

// ── VITA Group Tab switching ──────────────────────────────────────────────

function selectGroupTab(group) {
  vitaGroupTabs.forEach(tab => {
    const isActive = tab.dataset.group === group;
    tab.classList.toggle("active", isActive);
    tab.setAttribute("aria-selected", isActive ? "true" : "false");
  });
  presetGroups.forEach(pg => {
    const isActive = pg.id === `presetGroup${group.charAt(0).toUpperCase() + group.slice(1)}`;
    pg.hidden = !isActive;
    pg.classList.toggle("active", isActive);
  });
  intensityRow.hidden = (group === "custom");
}

vitaGroupTabs.forEach(tab => {
  tab.addEventListener("click", () => {
    selectGroupTab(tab.dataset.group);
  });
  tab.addEventListener("keydown", e => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      selectGroupTab(tab.dataset.group);
    }
  });
});

// ── Preset button selection ───────────────────────────────────────────────

function selectPreset(presetName) {
  activePreset = presetName;
  presetButtons.forEach(btn => {
    btn.classList.toggle("selected", btn.dataset.preset === presetName);
  });
  shadeBadge.textContent = presetName;
}

presetButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    if (!currentImageStem) return;
    selectPreset(btn.dataset.preset);
    requestShade(activePreset);
  });
});

// ── Intensity slider ──────────────────────────────────────────────────────

intensitySlider.addEventListener("input", () => {
  intensityValue.textContent = `${intensitySlider.value}%`;
  if (!currentImageStem) return;
  clearTimeout(shadeDebounceTimer);
  shadeDebounceTimer = setTimeout(() => requestShade(activePreset), 320);
});

// ── Custom sliders ────────────────────────────────────────────────────────

sliderBrightness.addEventListener("input", () => {
  outBrightness.textContent = sliderBrightness.value;
});
sliderYellow.addEventListener("input", () => {
  outYellow.textContent = `${sliderYellow.value}%`;
});
sliderWarmth.addEventListener("input", () => {
  outWarmth.textContent = sliderWarmth.value;
});
applyCustomBtn.addEventListener("click", () => {
  if (!currentImageStem) return;
  requestShadeCustom();
});

// ── Before / After toggle ─────────────────────────────────────────────────

beforeAfterBtn.addEventListener("click", () => {
  if (!currentShadeUrl) return;
  isShowingOriginal = !isShowingOriginal;
  beforeAfterBtn.setAttribute("aria-pressed", isShowingOriginal ? "true" : "false");
  beforeAfterBtn.classList.toggle("pressed", isShowingOriginal);

  if (isShowingOriginal) {
    shadeImage.src = `${currentUploadUrl}?t=${Date.now()}`;
    shadeImage.alt = "Original (before shade change)";
    beforeAfterBtn.classList.add("showing-before");
  } else {
    shadeImage.src = `${currentShadeUrl}?t=${Date.now()}`;
    shadeImage.alt = "Shade simulation result";
    beforeAfterBtn.classList.remove("showing-before");
  }
});

// ── API calls ─────────────────────────────────────────────────────────────

async function requestShade(presetName) {
  if (!currentImageStem) return;
  setShadeLoading(true);

  const intensity = parseInt(intensitySlider.value, 10) / 100;

  try {
    const response = await fetch("/shade", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_stem: currentImageStem,
        preset:     presetName,
        intensity:  intensity,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Shade failed.");

    applyShadeResult(data, presetName);
  } catch (err) {
    setShadeLoading(false);
    showErrorToast(`Shade error: ${err.message}`);
  }
}

async function requestShadeCustom() {
  if (!currentImageStem) return;
  setShadeLoading(true);

  try {
    const response = await fetch("/shade", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_stem:  currentImageStem,
        preset:      "custom",
        brightness:  parseFloat(sliderBrightness.value),
        yellow_red:  parseFloat(sliderYellow.value) / 100,
        warmth:      parseFloat(sliderWarmth.value),
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Shade failed.");

    applyShadeResult(data, "Custom");
  } catch (err) {
    setShadeLoading(false);
    showErrorToast(`Shade error: ${err.message}`);
  }
}

function applyShadeResult(data, labelOverride) {
  currentShadeUrl = data.shaded_url;
  isShowingOriginal = false;
  beforeAfterBtn.setAttribute("aria-pressed", "false");
  beforeAfterBtn.classList.remove("pressed", "showing-before");

  const newSrc = `${data.shaded_url}?t=${Date.now()}`;
  shadeImage.alt = `Shade: ${labelOverride || data.preset}`;
  shadeBadge.textContent = labelOverride || data.preset;
  shadeSavePath.textContent = `Saved: ${data.shade_path}`;

  downloadShadeBtn.href    = data.shaded_url;
  downloadShadeBtn.setAttribute("download", `shade_${data.preset}.jpg`);
  downloadShadeBtn.hidden  = false;

  shadeResultWrap.hidden = false;

  // Immediately hide spinner & update image
  setShadeLoading(false);
  shadeImage.src = newSrc;
}

function setShadeLoading(isLoading) {
  shadeLoading.hidden = !isLoading;
  shadeLoading.style.display = isLoading ? "flex" : "none";
  if (isLoading) {
    shadeResultWrap.hidden = false;
    shadeImage.style.opacity = "0.35";
  } else {
    shadeImage.style.opacity = "1";
  }
}

// ── Drop-zone events ──────────────────────────────────────────────────────

chooseButton.addEventListener("click", () => input.click());
form.addEventListener("click", event => {
  if (event.target !== chooseButton) input.click();
});
input.addEventListener("change", () => uploadFile(input.files[0]));

["dragenter", "dragover"].forEach(name => {
  form.addEventListener(name, event => {
    event.preventDefault();
    form.classList.add("dragging");
  });
});
["dragleave", "drop"].forEach(name => {
  form.addEventListener(name, event => {
    event.preventDefault();
    form.classList.remove("dragging");
  });
});
form.addEventListener("drop", event => {
  const file = event.dataTransfer?.files[0];
  if (file) uploadFile(file);
});

// ── Smile Restoration Panel ──────────────────────────────────────────────

const restorePanel = document.getElementById("restorePanel");
const restoreStep1 = document.getElementById("restoreStep1");
const analyzeBtn = document.getElementById("analyzeBtn");
const restoreAnalysis = document.getElementById("restoreAnalysis");
const analysisGrid = document.getElementById("analysisGrid");
const analysisDetails = document.getElementById("analysisDetails");

const restoreVlmInput = document.getElementById("restoreVlmInput");
const vlmJsonInput = document.getElementById("vlmJsonInput");
const loadSampleBtn = document.getElementById("loadSampleBtn");
const submitVlmBtn = document.getElementById("submitVlmBtn");

const restoreVlmDetails = document.getElementById("restoreVlmDetails");
const restorePlan = document.getElementById("restorePlan");
const planSummary = document.getElementById("planSummary");
const planEdits = document.getElementById("planEdits");

const restoreExecute = document.getElementById("restoreExecute");
const restoreBtn = document.getElementById("restoreBtn");
const restoreLoading = document.getElementById("restoreLoading");
const restoreLoadingText = document.getElementById("restoreLoadingText");

const restoreResult = document.getElementById("restoreResult");
const restoreComparisonImg = document.getElementById("restoreComparisonImg");
const downloadRestoredBtn = document.getElementById("downloadRestoredBtn");
const verificationBadge = document.getElementById("verificationBadge");
const generationMode = document.getElementById("generationMode");
const newVariationBtn = document.getElementById("newVariationBtn");
let variationSeed = 42;
let currentPlanId = null;
let currentAppearancePlan = null;
let designDraft = { selected_ids: null, arch_overrides: {}, replacements: [], excluded_regions: [] };
let designCanvasImage = null;
let activeJobId = null;
document.getElementById('cancelRestoreBtn').addEventListener('click',async () => {
  if (!activeJobId) return;
  const response=await fetch(`/jobs/${activeJobId}/cancel`,{method:'POST'});
  if (response.ok) restoreLoadingText.textContent='Cancellation requested; finishing the current render pass safely. Its result will be discarded.';
});
const designControls = ['designUpper','designLower','designShade','designShape','designAlignment','designAlignmentStrength'].map(id => document.getElementById(id));
const veneerControls = ['veneerTemplate','upperEdge','upperCurve','upperWidth','lowerEdge','lowerCurve','lowerWidth'].map(id=>document.getElementById(id));
const completionControl=document.getElementById('completeSmile');
completionControl.addEventListener('change',markDesignDirty);
function syncGeometryControls() {
  const inactive=designControls[4].value!=='aligned'||designControls[3].value==='original';
  const busy=!restoreLoading.hidden;
  veneerControls.forEach(control=>control.disabled=busy||inactive);
  designControls[5].disabled=busy||inactive;
}
designControls[4].addEventListener('change',syncGeometryControls);
designControls[3].addEventListener('change',syncGeometryControls);
veneerControls.forEach(control=>control.addEventListener('input',()=>{
  const output=document.getElementById(control.id+'Value');
  if(output) output.textContent=Math.round(Number(control.value)*100)+'%';
  markDesignDirty();
}));
const designCanvas = document.getElementById('designCanvas');

function readDesignOptions() {
  return { ...designDraft, upper_mode: designControls[0].value, lower_mode: designControls[1].value,
    shade: designControls[2].value, shape: designControls[3].value,
    alignment: designControls[4].value, alignment_strength: Number(designControls[5].value),
    veneer_template:veneerControls[0].value, complete_smile:completionControl.checked,
    arch_design:{upper:{edge_offset:Number(veneerControls[1].value),curve:Number(veneerControls[2].value),width:Number(veneerControls[3].value)},
                 lower:{edge_offset:Number(veneerControls[4].value),curve:Number(veneerControls[5].value),width:Number(veneerControls[6].value)}} };
}

function markDesignDirty() {
  currentPlanId = null;
  restoreBtn.disabled = true;
  restoreResult.hidden = true;
  document.getElementById('designDirtyNotice').hidden = false;
}

designControls.forEach((control,index) => control.addEventListener('change', () => {
  if (index < 2) designDraft.selected_ids = null;
  markDesignDirty();
}));
designControls[5].addEventListener('input',() => {
  document.getElementById('alignmentStrengthValue').textContent=Math.round(Number(designControls[5].value)*100)+'%';
  markDesignDirty();
});
document.getElementById('updateDesignBtn').addEventListener('click', () => analyzeBtn.click());
document.getElementById('clearDesignRegions').addEventListener('click', () => {
  designDraft.replacements = [];
  designDraft.excluded_regions = [];
  markDesignDirty();
  drawDesignCanvas();
});

function drawDesignCanvas() {
  const ctx = designCanvas.getContext('2d');
  ctx.clearRect(0,0,512,512);
  if (designCanvasImage) ctx.drawImage(designCanvasImage,0,0,512,512);
  if (!currentAppearancePlan) return;
  const [x1,y1,x2,y2] = currentAppearancePlan.crop_box;
  for (const [regions,color] of [[designDraft.excluded_regions,'#f87171'],[designDraft.replacements,'#34d399']]) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    for (const region of regions) {
      ctx.beginPath();
      region.polygon.forEach(([x,y],index) => {
        const px=(x-x1)*512/(x2-x1), py=(y-y1)*512/(y2-y1);
        if (index) ctx.lineTo(px,py); else ctx.moveTo(px,py);
      });
      ctx.closePath(); ctx.stroke();
    }
  }
  document.getElementById('designRegionSummary').textContent =
    `${designDraft.excluded_regions.length} existing mask corrections`;
}

let designDragStart = null;
function canvasPoint(event) {
  const box=designCanvas.getBoundingClientRect();
  return [Math.max(0,Math.min(511,(event.clientX-box.left)*512/box.width)),
          Math.max(0,Math.min(511,(event.clientY-box.top)*512/box.height))];
}
designCanvas.addEventListener('pointerdown',event => {
  if (!currentAppearancePlan || restoreLoading.hidden === false) return;
  designDragStart=canvasPoint(event); designCanvas.setPointerCapture(event.pointerId);
});
designCanvas.addEventListener('pointermove',event => {
  if (!designDragStart) return;
  drawDesignCanvas();
  const end=canvasPoint(event),ctx=designCanvas.getContext('2d');
  ctx.strokeStyle='#fbbf24';ctx.lineWidth=2;
  ctx.strokeRect(designDragStart[0],designDragStart[1],end[0]-designDragStart[0],end[1]-designDragStart[1]);
});
designCanvas.addEventListener('pointerup',event => {
  if (!designDragStart || !currentAppearancePlan) return;
  const end=canvasPoint(event),start=designDragStart;designDragStart=null;
  const left=Math.min(start[0],end[0]),right=Math.max(start[0],end[0]);
  const top=Math.min(start[1],end[1]),bottom=Math.max(start[1],end[1]);
  if (right-left<5 || bottom-top<5) { drawDesignCanvas(); return; }
  const mode=document.getElementById('designDrawMode').value;
  const radius=mode==='exclude' ? 0 : Math.min((right-left)*.16,(bottom-top)*.08);
  const local=[[left+radius,top],[right-radius,top],[right,top+radius],[right,bottom-radius],
    [right-radius,bottom],[left+radius,bottom],[left,bottom-radius],[left,top+radius]];
  const [x1,y1,x2,y2]=currentAppearancePlan.crop_box;
  const polygon=local.map(([x,y]) => [x1+x*(x2-x1)/512,y1+y*(y2-y1)/512]);
  if (mode==='exclude') designDraft.excluded_regions.push({polygon});
  else designDraft.replacements.push({polygon,arch:mode.endsWith('upper')?'upper':'lower',reviewed:true});
  markDesignDirty();drawDesignCanvas();
});
designCanvas.addEventListener('pointercancel',() => { designDragStart=null;drawDesignCanvas(); });

function renderDesignPlan(data) {
  currentPlanId=data.plan_id;
  currentAppearancePlan=data.appearance_plan || null;
  const quality=document.getElementById('inputQualityNotice');
  quality.textContent=(data.input_quality?.warnings || []).join(' ');
  quality.hidden=!quality.textContent;
  document.getElementById('designDirtyNotice').hidden=true;
  if (!currentAppearancePlan) return;
  const options=currentAppearancePlan.options;
  const proposed=currentAppearancePlan.crowns.filter(c=>c.presence==='proposed_smile_tooth').length;
  document.getElementById('completionNotice').textContent=proposed ? `${proposed} proposed teeth added to visible gaps. Review their shapes and count in the outline preview.` : 'No additional teeth proposed. Completion uses tooth-sized spaces between visible neighboring teeth.';
  designDraft={selected_ids:options.selected_ids,arch_overrides:{...options.arch_overrides},
    replacements:options.replacements.slice(),excluded_regions:options.excluded_regions.slice(),tooth_type_overrides:{...options.tooth_type_overrides}};
  const choices=document.getElementById('designToothChoices');choices.replaceChildren();
  for (const crown of currentAppearancePlan.crowns.filter(c => c.presence==='observed')) {
    const label=document.createElement('label'),check=document.createElement('input'),text=document.createElement('span');
    check.type='checkbox';check.checked=crown.operation!=='preserve';check.dataset.toothId=crown.tooth_id;
    text.textContent=`${crown.tooth_id} · ${crown.tooth_type}`;
    const arch=document.createElement('select');arch.setAttribute('aria-label',`Arch for ${crown.tooth_id}`);
    for (const value of ['upper','lower']) {
      const option=document.createElement('option');option.value=value;option.textContent=value;arch.appendChild(option);
    }
    arch.value=crown.arch;
    check.addEventListener('change',() => {
      designDraft.selected_ids=Array.from(choices.querySelectorAll('input:checked')).map(el=>el.dataset.toothId);
      markDesignDirty();
    });
    arch.addEventListener('change',() => { designDraft.arch_overrides[crown.tooth_id]=arch.value;markDesignDirty(); });
    const type=document.createElement('select');type.setAttribute('aria-label',`Tooth type for ${crown.tooth_id}`);
    for(const value of ['central','lateral','canine','premolar','unknown']) {
      const option=document.createElement('option');option.value=value;option.textContent=value;type.appendChild(option);
    }
    type.value=crown.tooth_type;
    type.addEventListener('change',()=>{designDraft.tooth_type_overrides[crown.tooth_id]=type.value;markDesignDirty();});
    label.append(check,text,arch,type);choices.appendChild(label);
  }
  const image=new Image(),renderedId=data.plan_id;
  image.onload=() => { if (currentAppearancePlan?.plan_id===renderedId) { designCanvasImage=image;drawDesignCanvas(); } };
  image.src=data.selection_url || data.edit_area_url;
}
newVariationBtn.addEventListener('click', () => {
  try {
    const settings = JSON.parse(vlmJsonInput.value || '{}');
    variationSeed = (Number(settings.seed ?? variationSeed) + 1009) % 4294967295;
    settings.seed = variationSeed;
    vlmJsonInput.value = JSON.stringify(settings, null, 2);
    restoreBtn.click();
  } catch (error) {
    showErrorToast('Please correct the advanced settings JSON before generating another variation.');
  }
});

function resetRestorePanel() {
  restoreStep1.hidden = false;
  restoreAnalysis.hidden = true;
  if (restoreVlmDetails) restoreVlmDetails.hidden = true;
  restorePlan.hidden = true;
  restoreExecute.hidden = true;
  restoreResult.hidden = true;
  setRestoreLoading(false);
  vlmJsonInput.value = "";
  restoreComparisonImg.src = "";
  downloadRestoredBtn.href = "#";
  variationSeed = 42;
  currentPlanId = null;
  currentAppearancePlan = null;
  designDraft = { selected_ids: null, arch_overrides: {}, replacements: [], excluded_regions: [] };
  designCanvasImage = null;
  designControls[0].value='veneer_preview';designControls[1].value='veneer_preview';
  designControls[2].value='natural';designControls[3].value='balanced';
  designControls[4].value='aligned';designControls[5].value='0.85';
  completionControl.checked=true;
  veneerControls.forEach((control,i)=>{control.value=i===0?'legacy':control.id.endsWith('Width')?'1':'0';
    const output=document.getElementById(control.id+'Value');if(output)output.textContent=control.id.endsWith('Width')?'100%':'0%';});
  document.getElementById('inputQualityNotice').hidden=true;
  document.getElementById('alignmentStrengthValue').textContent='85%';
  document.getElementById('designToothChoices').replaceChildren();
  document.getElementById('designDirtyNotice').hidden=true;
  downloadRestoredBtn.hidden=false;
  generationMode.value = 'fast';
  document.getElementById('candidateChoices')?.remove();
  syncGeometryControls();
}

function setRestoreLoading(isLoading, text = "Processing...") {
  restoreLoading.hidden = !isLoading;
  restoreLoadingText.textContent = text;
  document.getElementById('cancelRestoreBtn').hidden=!isLoading || !activeJobId;
  completionControl.disabled=isLoading;
  
  const buttonsToDisable = [analyzeBtn, submitVlmBtn, restoreBtn, newVariationBtn, generationMode, chooseButton, input,
    ...designControls,...veneerControls,document.getElementById('updateDesignBtn'),document.getElementById('clearDesignRegions')];
  buttonsToDisable.forEach(btn => {
    if (btn) btn.disabled = isLoading;
  });
  restoreBtn.disabled=isLoading || !currentPlanId;
  document.querySelectorAll('#designToothChoices input,#designToothChoices select').forEach(el=>el.disabled=isLoading);
  syncGeometryControls();
}

function escapeText(value) {
  const element = document.createElement('span');
  element.textContent = String(value ?? '');
  return element.innerHTML;
}

function renderPlan(proposedEdits, numEdits, status) {
  const count = numEdits || (proposedEdits ? proposedEdits.length : 0);
  planSummary.innerHTML = `
    <div style="display:flex; align-items:center; justify-content:space-between;">
      <strong>${Number(count)} Cosmetic Preview Area${count === 1 ? '' : 's'}</strong>
      <span style="font-size:12px; color:var(--muted);">Status: ${escapeText(status || 'Ready')}</span>
    </div>
    <p style="margin-top:6px; font-size:13px; color:var(--text);">Full-smile appearance preview. Review the highlighted area and final tooth shapes; this is not a treatment plan.</p>
  `;

  if (proposedEdits && proposedEdits.length > 0) {
    planEdits.innerHTML = proposedEdits.map((edit, idx) => {
      const toothLabel = escapeText(edit.fdi_number ? `Tooth #${edit.fdi_number}` : (edit.tooth_id || `Tooth ${idx+1}`));
      const editClass = String(edit.edit_type || 'default').replace(/[^a-z_]/g, '');
      const typeLabel = escapeText((edit.edit_type || 'edit').replace(/_/g, ' ').toUpperCase());
      const desc = escapeText(edit.description || edit.brushnet_prompt || 'Dental restoration inpainting');
      const priority = escapeText(edit.priority || 'medium');
      return `
        <div class="edit-card">
          <div class="edit-card-header">
            <div class="edit-tooth-title">${toothLabel} <span style="font-weight:400; color:var(--muted); font-size:12px;">(${escapeText(edit.tooth_type || 'tooth')})</span></div>
            <div class="edit-badges">
              <span class="edit-badge ${editClass}">${typeLabel}</span>
              <span class="edit-priority">${priority}</span>
            </div>
          </div>
          <div class="edit-card-desc">${desc}</div>
        </div>
      `;
    }).join('');
  } else {
    planEdits.innerHTML = `<div style="padding:12px; color:var(--muted); font-size:13px;">No edits needed based on this analysis.</div>`;
  }
}

analyzeBtn.addEventListener("click", async () => {
  if (!currentImageStem) return;
  setRestoreLoading(true, "Measuring the smile and preparing the cosmetic preview...");
  const requestStem = currentImageStem;
  
  try {
    const response = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_stem: currentImageStem, design_options: readDesignOptions() })
    });
    
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error || "Analysis failed");
    }
    const data = await response.json();
    if (requestStem !== currentImageStem) return;
    
    const smile = data.geometry?.smile || {};
    const teeth = data.geometry?.teeth || [];
    const arcType = smile.arc_type ? smile.arc_type.charAt(0).toUpperCase() + smile.arc_type.slice(1) : "Consonant";
    const midlineCant = smile.midline_cant_deg !== undefined ? `${smile.midline_cant_deg.toFixed(1)}°` : "0.0°";
    const midlineStatus = smile.midline_assessment || "normal";
    const goldenScore = smile.golden_pct_score !== undefined ? `${Math.round(smile.golden_pct_score)}%` : "N/A";
    const toothCount = teeth.length || data.num_instances || 0;
    const upperCount = smile.num_teeth_upper || toothCount;
    const lowerCount = smile.num_teeth_lower || 0;

    analysisGrid.innerHTML = `
      <div class="stat-card"><strong>${escapeText(arcType)}</strong><span>Estimated tooth-edge curve</span></div>
      <div class="stat-card"><strong>${toothCount}</strong><span>Teeth Detected (${upperCount}U / ${lowerCount}L)</span></div>
      <div class="stat-card"><strong>${midlineCant}</strong><span>Tooth-line tilt (approximate)</span></div>
      <div class="stat-card"><strong>Patient-specific</strong><span>Editable crown proportions</span></div>
    `;

    if (teeth.length > 0) {
      const toothListHtml = teeth.slice(0, 8).map(t => {
        const fdi = escapeText(t.fdi_number ? `#${t.fdi_number}` : t.tooth_id);
        const type = escapeText(t.tooth_type || 'tooth');
        const wh = t.wh_ratio ? `W/H: ${t.wh_ratio.toFixed(2)}` : '';
        const rot = t.rotation_assessment && t.rotation_assessment !== 'normal' ? `• Rot: ${escapeText(t.rotation_assessment)}` : '';
        return `<span style="display:inline-block; margin:2px 4px; padding:3px 8px; background:#f1f5f9; border-radius:4px; font-size:11.5px; font-weight:600;">${fdi} (${type}) ${wh} ${rot}</span>`;
      }).join('');
      analysisDetails.innerHTML = `<div style="margin-top:10px;"><strong style="font-size:12px; color:var(--muted); text-transform:uppercase;">Segmented Landmarks:</strong><div style="margin-top:4px;">${toothListHtml}</div></div>`;
    } else {
      analysisDetails.innerHTML = '';
    }

    // Auto-populate VLM JSON editor with the synthesized clinical findings
    if (data.vlm_analysis) {
      vlmJsonInput.value = JSON.stringify(data.vlm_analysis, null, 2);
    }
    
    // Automatically render the clinical restoration plan
    renderPlan(data.proposed_edits, data.num_edits, data.status);
    renderDesignPlan(data);
    if (data.edit_area_url) {
      const preview = document.createElement('img');
      preview.src = data.edit_area_url;
      preview.alt = 'Highlighted planned edit area for the full-smile cosmetic preview';
      preview.style.cssText = 'max-width:280px;width:100%;border-radius:8px;margin-top:12px';
      analysisDetails.appendChild(preview);
    }
    
    restoreStep1.hidden = true;
    restoreAnalysis.hidden = false;
    restorePlan.hidden = false;
    restoreExecute.hidden = false;
    if (restoreVlmDetails) restoreVlmDetails.hidden = false;
  } catch (error) {
    showErrorToast(error.message);
  } finally {
    setRestoreLoading(false);
  }
});

loadSampleBtn.addEventListener("click", (e) => {
  e.preventDefault();
  vlmJsonInput.value = JSON.stringify({
    smile_analysis: { source: 'user_appearance_preference', confidence: 0 },
    teeth: [], appearance_request: readDesignOptions(),
    steps: 32, guidance_scale: 7.5, brushnet_conditioning_scale: 0.58,
    control_guidance_end: 0.74, conditioning_mode: 'scaffold', seed: 42
  }, null, 2);
});

submitVlmBtn.addEventListener("click", async () => {
  if (!currentImageStem) return;
  
  let parsedJson;
  try {
    parsedJson = JSON.parse(vlmJsonInput.value);
  } catch (e) {
    showErrorToast("Invalid JSON syntax: " + e.message);
    return;
  }
  
  setRestoreLoading(true, "Updating restoration plan with custom JSON...");
  
  try {
    const response = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_stem: currentImageStem, manual_analysis: parsedJson, design_options: readDesignOptions() })
    });
    
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error || (err.validation_errors ? err.validation_errors.join("; ") : "Analysis failed"));
    }
    const data = await response.json();
    
    renderPlan(data.proposed_edits, data.num_edits, data.status);
    renderDesignPlan(data);
    
    restorePlan.hidden = false;
    restoreExecute.hidden = false;
  } catch (error) {
    showErrorToast(error.message);
  } finally {
    setRestoreLoading(false);
  }
});

restoreBtn.addEventListener("click", async () => {
  if (!currentImageStem) return;
  if (!currentPlanId) { showErrorToast('Update the design preview before generating.'); return; }
  setRestoreLoading(true, "Generating your smile preview with BrushNet...");
  const requestStem = currentImageStem;
  activeJobId=Array.from(crypto.getRandomValues(new Uint8Array(16))).map(v=>v.toString(16).padStart(2,'0')).join('');
  document.getElementById('cancelRestoreBtn').hidden=false;
  const thisJob=activeJobId;
  const progressTimer=setInterval(async () => {
    if (activeJobId!==thisJob) return;
    try {
      const response=await fetch(`/jobs/${thisJob}`);
      if (!response.ok) return;
      const record=await response.json();
      if (record.cancel_requested) return;
      if (record.status==='running') {
        const progress=record.progress || {};
        restoreLoadingText.textContent=progress.step ? `${progress.phase} crowns: step ${progress.step} of ${progress.total_steps}` : 'Loading the restoration model...';
      }
    } catch (_) { /* The original generation request reports any failure. */ }
  },1500);
  
  try {
    const payload = { image_stem: requestStem, plan_id: currentPlanId, job_id: thisJob, quality_mode: generationMode.value, seed: variationSeed };
    
    // Read custom prompts and diffusion parameters from VLM JSON editor if present
    if (vlmJsonInput && vlmJsonInput.value) {
      try {
        const parsed = JSON.parse(vlmJsonInput.value);
        if (parsed.positive_prompt) {
          if (typeof parsed.positive_prompt === 'string') {
            payload.prompt = parsed.positive_prompt;
          } else if (typeof parsed.positive_prompt === 'object') {
            const parts = [];
            if (parsed.positive_prompt.core_transformation) parts.push(parsed.positive_prompt.core_transformation);
            if (Array.isArray(parsed.positive_prompt.structural_adjustments)) parts.push(...parsed.positive_prompt.structural_adjustments);
            if (Array.isArray(parsed.positive_prompt.material_and_texture)) parts.push(...parsed.positive_prompt.material_and_texture);
            if (parts.length > 0) payload.prompt = parts.join(", ");
          }
        } else if (parsed.prompt) {
          payload.prompt = parsed.prompt;
        }
        if (parsed.negative_prompt) {
          payload.negative_prompt = typeof parsed.negative_prompt === 'object' && Array.isArray(parsed.negative_prompt.exclusions) ?
            parsed.negative_prompt.exclusions.join(", ") : String(parsed.negative_prompt);
        }
        if (parsed.brushnet_strength !== undefined || parsed.brushnet_conditioning_scale !== undefined) {
          payload.brushnet_conditioning_scale = parsed.brushnet_conditioning_scale ?? parsed.brushnet_strength;
        }
        if (parsed.guidance_scale !== undefined) payload.guidance_scale = parsed.guidance_scale;
        if (parsed.control_guidance_end !== undefined) payload.control_guidance_end = parsed.control_guidance_end;
        if (parsed.seed !== undefined) payload.seed = parsed.seed;
        if (parsed.conditioning_mode !== undefined) payload.conditioning_mode = parsed.conditioning_mode;
        if (parsed.steps) payload.steps = parsed.steps;
        if (parsed.quality_mode) payload.quality_mode = parsed.quality_mode;
        if (parsed.force_bilateral_mirror !== undefined) payload.force_bilateral_mirror = parsed.force_bilateral_mirror;
      } catch (e) {
        // Keep default payload on parse error
      }
    }

    const response = await fetch("/restore", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    
    const data = await response.json().catch(() => ({}));
    if (requestStem !== currentImageStem) return;
    if (!response.ok || data.error) {
      throw new Error(data.error || "Restoration failed on server.");
    }

    if (!data.comparison_url && !data.restored_url) {
      throw new Error("No restored image URL was generated by the pipeline.");
    }
    
    const cacheBuster = `?t=${Date.now()}`;
    const displayUrl = (data.comparison_url || data.restored_url) + cacheBuster;
    const downloadUrl = (data.restored_url || data.comparison_url) + "?download=1";
    
    restoreComparisonImg.src = displayUrl;
    downloadRestoredBtn.href = downloadUrl;
    downloadRestoredBtn.setAttribute("download", data.restored_filename || "smile_restored.jpg");
    downloadRestoredBtn.hidden=data.passed===false || data.verification?.passed===false;
    document.getElementById('candidateChoices')?.remove();
    if (data.candidates?.length > 1) {
      const choices = document.createElement('div');
      choices.id = 'candidateChoices';
      choices.textContent = 'Compare both results and choose the more natural smile: ';
      data.candidates.forEach((candidate, index) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'btn btn-secondary';
        button.textContent = `Result ${index + 1}${candidate.passed===false || candidate.verification?.passed===false?' — needs revision':''}`;
        button.addEventListener('click', () => {
          downloadRestoredBtn.hidden=candidate.passed===false || candidate.verification?.passed===false;
          restoreComparisonImg.src = candidate.comparison_url;
          downloadRestoredBtn.href = candidate.restored_url + '?download=1';
          downloadRestoredBtn.setAttribute('download', candidate.restored_filename);
          const fidelity=document.getElementById('fidelityNotice');
          if(fidelity) fidelity.textContent=(candidate.verification?.metrics?.design_fidelity?.warnings || []).join(' ') || 'Review veneer shape, separation and margins visually; automatic checks do not validate anatomy.';
          verificationBadge.textContent = candidate.verification.passed
            ? 'Technical checks passed — review the smile visually'
            : 'Technical issues — review required';
          verificationBadge.className = 'verification-badge ' + (candidate.verification.passed ? 'passed' : 'failed');
          document.getElementById('verificationMetrics')?.replaceChildren(
            document.createTextNode(`Result ${index+1} · ${candidate.engine_type} · seed ${candidate.seed}. Visual review required.`));
        });
        choices.appendChild(button);
      });
      restoreResult.appendChild(choices);
    }
    
    const ssimVal = data.verification?.metrics?.unmasked_ssim;
    const ssimText = (ssimVal !== undefined && ssimVal !== null) ? ` (SSIM: ${(ssimVal * 100).toFixed(2)}%)` : '';
    const passed = Boolean(data.passed);
    
    verificationBadge.textContent = (passed ? "Technical checks passed — review the smile visually" : "Technical issues — review required");
    verificationBadge.className = "verification-badge " + (passed ? "passed" : "failed");
    
    let metricsContainer = document.getElementById("verificationMetrics");
    if (!metricsContainer) {
      metricsContainer = document.createElement("div");
      metricsContainer.id = "verificationMetrics";
      metricsContainer.className = "verification-metrics";
      restoreResult.appendChild(metricsContainer);
    }
    
    const m = data.verification?.metrics || {};
    let fidelityNotice=document.getElementById('fidelityNotice');
    if(!fidelityNotice){fidelityNotice=document.createElement('p');fidelityNotice.id='fidelityNotice';restoreResult.appendChild(fidelityNotice);}
    fidelityNotice.textContent=(m.design_fidelity?.warnings || []).join(' ') || 'Review veneer shape, separation and margins visually; automatic checks do not validate anatomy.';
    metricsContainer.innerHTML = `
      <div class="metric-item">Changes outside edit area: <strong>${m.protected_pixels_changed ?? 'Not checked'} pixels</strong></div>
      <div class="metric-item">Restorative Inpaint: <strong>${m.edit_happened ? 'Applied' : 'No-op'}</strong></div>
      <div class="metric-item">Gingival Bleed Prevention: <strong>${m.no_color_bleed !== false ? 'Protected' : 'Warning'}</strong></div>
      <div class="metric-item">Pixel Artifact Check: <strong>${m.no_artifacts !== false ? 'Clean' : 'Artifacts detected'}</strong></div>
      <div class="metric-item">Generator: <strong>${data.restore_meta?.engine_type === 'brushnet' ? 'BrushNet' : data.restore_meta?.engine_type === 'sd_inpaint' ? 'SD inpainting fallback' : data.restore_meta?.engine_type === 'deterministic_shade' ? 'Shade adjustment — original geometry' : 'Color-only fallback — no neural restoration'}</strong></div>
      <div class="metric-item">Pipeline Latency: <strong>${data.runtime_seconds || 0}s</strong></div>
    `;
    
    restoreExecute.hidden = true;
    restoreResult.hidden = false;
  } catch (error) {
    showErrorToast(error.message);
  } finally {
    clearInterval(progressTimer);
    activeJobId=null;
    setRestoreLoading(false);
  }
});
