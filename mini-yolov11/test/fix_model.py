"""
Script to remove YOLOE class from model.py
"""
with open('d:\\Project\\yolo-v11-rebuild\\mini-yolov11\\ultralytics\\models\\yolo\\model.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find YOLOE class start
start_idx = None
for i, line in enumerate(lines):
    if line.strip().startswith('class YOLOE(Model):'):
        start_idx = i
        break

if start_idx is None:
    print("YOLOE class not found")
else:
    # Find the end of YOLOE class (next class definition or end of file)
    end_idx = None
    for i in range(start_idx + 1, len(lines)):
        line = lines[i].strip()
        if line.startswith('class ') and '(' in line:
            end_idx = i
            break

    if end_idx is None:
        end_idx = len(lines)

    # Remove YOLOE class
    new_lines = lines[:start_idx] + lines[end_idx:]

    with open('d:\\Project\\yolo-v11-rebuild\\mini-yolov11\\ultralytics\\models\\yolo\\model.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"Removed YOLOE class (lines {start_idx+1} to {end_idx})")
