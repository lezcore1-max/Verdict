import os

with open("frontend/app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

try:
    idx_start = next(i for i, l in enumerate(lines) if l.startswith('paper_id = st.session_state.get('))
    idx_end = next(i for i, l in enumerate(lines) if l.startswith('        st.rerun()') and i > idx_start) + 1

    new_lines = lines[:idx_start] + ['def main():\n'] + ['    ' + l if l.strip() else l for l in lines[idx_start:idx_end]] + ['\n'] + lines[idx_end:] + ['\nif __name__ == "__main__":\n    main()\n']
    
    with open("frontend/app.py", "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print("Fixed app.py successfully!")
except Exception as e:
    print(f"Error: {e}")
