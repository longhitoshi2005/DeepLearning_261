import os, re

def check_doc(path):
    print(f'=== CHECKING {path} ===')
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Find all cite keys
    cites = set()
    for m in re.finditer(r'\\cite\{([^}]+)\}', text):
        keys = [k.strip() for k in m.group(1).split(',')]
        cites.update(keys)

    # Find all bibitem keys
    bibitems = set(re.findall(r'\\bibitem\{([^}]+)\}', text))

    print(f'Total cite keys used: {len(cites)}')
    print(f'Total bibitems defined: {len(bibitems)}')

    missing = cites - bibitems
    unused = bibitems - cites

    if missing:
        print(f'CRITICAL ERROR - MISSING BIBITEMS FOR: {missing}')
    else:
        print('ALL CITATION KEYS ARE PROPERLY DEFINED!')

    if unused:
        print(f'Unused bibitems (info only): {unused}')
    print()

p_vi = 'main_vi.tex' if os.path.exists('main_vi.tex') else 'report/main_vi.tex'
p_en = 'main.tex' if os.path.exists('main.tex') else 'report/main.tex'
p_sl = 'slides.tex' if os.path.exists('slides.tex') else 'report/slides.tex'

check_doc(p_vi)
check_doc(p_en)
check_doc(p_sl)
