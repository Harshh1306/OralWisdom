"""Short operation-specific rendering prompts from the canonical plan."""
def arch_prompt(plan,arch):
    crowns=[c for c in plan.get('crowns',[]) if c['arch']==arch and c['operation'] not in ('preserve','shade_only')]
    replacement=any(c['operation']=='replacement_crown_preview' for c in crowns)
    shade=plan.get('options',{}).get('shade','natural')
    color={'natural':'natural warm ivory','brighter':'bright natural ivory','bright':'bright ivory ceramic'}[shade]
    form='narrow individually separated lower crowns' if arch=='lower' else 'distinct central incisors, smaller lateral incisors, natural canines'
    operation='replacement crown matching adjacent teeth' if replacement else 'cosmetic ceramic veneer restoration'
    if any(c.get('presence')=='proposed_smile_tooth' for c in crowns):
        operation='complete cosmetic smile, proposed teeth matching neighboring enamel'
    return (f'{operation}, {form}, {color}, rounded incisal corners, realistic enamel texture, '
            'subtle translucency, restrained highlights matching original photograph, natural contacts, preserved gum margin')
