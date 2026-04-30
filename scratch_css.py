import re

with open('frontend/index.css', 'r', encoding='utf-8') as f:
    css = f.read()

# 1. Update variables
css = re.sub(r'--bg-base:.*?;', '--bg-base: #f4f4f4;', css)
css = re.sub(r'--bg-surface:.*?;', '--bg-surface: #ffffff;', css)
css = re.sub(r'--bg-card:.*?;', '--bg-card: #ffffff;', css)
css = re.sub(r'--bg-hover:.*?;', '--bg-hover: #e5e5e5;', css)
css = re.sub(r'--border:.*?;', '--border: #d4d4d4;', css)
css = re.sub(r'--border-bright:.*?;', '--border-bright: #a3a3a3;', css)
css = re.sub(r'--text-primary:.*?;', '--text-primary: #171717;', css)
css = re.sub(r'--text-secondary:.*?;', '--text-secondary: #404040;', css)
css = re.sub(r'--text-muted:.*?;', '--text-muted: #737373;', css)

css = re.sub(r'--accent-indigo:.*?;', '--accent-indigo: #000000;', css)
css = re.sub(r'--accent-cyan:.*?;', '--accent-cyan: #262626;', css)
css = re.sub(r'--accent-emerald:.*?;', '--accent-emerald: #171717;', css)
css = re.sub(r'--accent-amber:.*?;', '--accent-amber: #404040;', css)
css = re.sub(r'--accent-rose:.*?;', '--accent-rose: #000000;', css)

css = re.sub(r'--gradient-brand:.*?;', '--gradient-brand: linear-gradient(135deg, #000000 0%, #404040 100%);', css)
css = re.sub(r'--gradient-warm:.*?;', '--gradient-warm: linear-gradient(135deg, #262626 0%, #737373 100%);', css)

css = re.sub(r'--shadow-glow:.*?;', '--shadow-glow: 0 4px 20px rgba(0, 0, 0, 0.05);', css)

# 2. Orbs
css = css.replace('background: #6366f1;', 'background: #e5e5e5;')
css = css.replace('background: #06b6d4;', 'background: #d4d4d4;')
css = css.replace('background: #8b5cf6;', 'background: #a3a3a3;')

# 3. Call Banner Widget
css = re.sub(r'background: linear-gradient\(135deg, rgba\(22, 22, 40, 0\.8\) 0%, rgba\(10, 10, 18, 0\.9\) 100%\);', 
             'background: #ffffff;', css)
css = re.sub(r'box-shadow: 0 10px 30px rgba\(0, 0, 0, 0\.3\), 0 0 40px rgba\(99, 102, 241, 0\.15\);', 
             'box-shadow: 0 8px 30px rgba(0, 0, 0, 0.05);', css)
css = re.sub(r'box-shadow: 0 0 0 0 rgba\(99, 102, 241, 0\.4\)', 'box-shadow: 0 0 0 0 rgba(0, 0, 0, 0.2)', css)
css = re.sub(r'box-shadow: 0 0 0 15px rgba\(99, 102, 241, 0\)', 'box-shadow: 0 0 0 15px rgba(0, 0, 0, 0)', css)
css = re.sub(r'box-shadow: 0 0 0 0 rgba\(99, 102, 241, 0\)', 'box-shadow: 0 0 0 0 rgba(0, 0, 0, 0)', css)
css = re.sub(r'rgba\(99, 102, 241, 0\.15\)', 'rgba(0, 0, 0, 0.05)', css)

# 4. Navbar & Glass Card specific
css = re.sub(r'background: rgba\(10, 10, 18, 0\.8\);', 'background: rgba(255, 255, 255, 0.9);', css)
css = re.sub(r'background: rgba\(10,10,18,0\.6\);', 'background: #ffffff;', css) # inputs
css = re.sub(r'background: rgba\(10,10,18,0\.5\);', 'background: #f4f4f4;', css) # outputs

# 5. Badges
css = re.sub(r'rgba\(16,185,129,0\.15\)', '#f4f4f4', css)
css = re.sub(r'rgba\(16,185,129,0\.3\)', '#d4d4d4', css)
css = re.sub(r'rgba\(245,158,11,0\.15\)', '#f4f4f4', css)
css = re.sub(r'rgba\(245,158,11,0\.3\)', '#d4d4d4', css)
css = re.sub(r'rgba\(244,63,94,0\.15\)', '#f4f4f4', css)
css = re.sub(r'rgba\(244,63,94,0\.3\)', '#d4d4d4', css)
css = re.sub(r'rgba\(244,63,94,0\.25\)', '#ffffff', css)
css = re.sub(r'rgba\(244,63,94,0\.5\)', '#000000', css)
css = re.sub(r'#ff4466', '#000000', css)

# 6. Specific hardcoded colors
css = re.sub(r'rgba\(99,102,241,0\.12\)', 'rgba(0,0,0,0.04)', css)
css = re.sub(r'rgba\(99,102,241,0\.08\)', 'rgba(0,0,0,0.03)', css)
css = re.sub(r'rgba\(99,102,241,0\.15\)', 'rgba(0,0,0,0.05)', css)
css = re.sub(r'rgba\(99,102,241,0\.05\)', 'rgba(0,0,0,0.02)', css)
css = re.sub(r'rgba\(99,102,241,0\.25\)', 'rgba(0,0,0,0.1)', css)
css = re.sub(r'rgba\(6,182,212,0\.06\)', 'rgba(0,0,0,0.03)', css)
css = re.sub(r'rgba\(255,255,255,0\.08\)', 'rgba(0,0,0,0.1)', css)
css = re.sub(r'rgba\(244,63,94,0\.1\)', 'rgba(0,0,0,0.05)', css)
css = re.sub(r'color: white;', 'color: #ffffff;', css)
css = re.sub(r'box-shadow: 0 0 20px rgba\(99,102,241,0\.15\);', 'box-shadow: 0 4px 12px rgba(0,0,0,0.1);', css)

with open('frontend/index.css', 'w', encoding='utf-8') as f:
    f.write(css)

print("CSS updated successfully")
