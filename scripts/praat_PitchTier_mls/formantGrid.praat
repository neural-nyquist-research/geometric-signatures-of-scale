# --- 1. 路径与环境配置 ---
# 默认处理脚本所在目录下的所有 .wav 文件
current_dir$ = "./"

# 扫描当前文件夹中的所有 .wav 文件
Create Strings as file list: "fileList", current_dir$ + "*.wav"
num_files = Get number of strings

if num_files == 0
    exitScript: "错误：在当前文件夹中未发现 .wav 文件！"
endif

writeInfoLine: ">>> 批量共振峰提取开始，总文件数: ", num_files

# --- 2. 循环处理逻辑 ---
for i from 1 to num_files
    selectObject: "Strings fileList"
    file_name$ = Get string: i
    
    # A. 读取音频
    Read from file: current_dir$ + file_name$
    sound_id = selected("Sound")
    sound_name$ = selected$("Sound")
    
    appendInfoLine: "正在处理音频: ", file_name$
    
    # B. 提取共振峰 (Formant Extraction)
    # 参数说明：
    # 0.1: Time step (秒) -> 严格对应 100ms 步长
    # 5: Max number of formants (通常提取 F1-F3，设为 5 比较稳健)
    # 5500: Maximum formant (Hz) -> 成人女性/通用标准为 5500
    # 0.025: Window length (秒)
    # 50: Pre-emphasis from (Hz)
    To Formant (burg): 0.1, 5, 5500, 0.025, 50
    formant_id = selected("Formant")
    
    # C. 转换为 FormantGrid (便于后续解析 F1/F2 数值)
    Down to FormantGrid
    grid_id = selected("FormantGrid")
    
    # D. 保存为 .FormantGrid 文本文件
    output_file$ = current_dir$ + sound_name$ + ".FormantGrid"
    Save as text file: output_file$
    
    # E. 内存清理 (非常关键，防止处理 1 小时音频时内存溢出)
    selectObject: sound_id
    plusObject: formant_id
    plusObject: grid_id
    Remove
    
    appendInfoLine: "成功生成: ", output_file$
endfor

# 最终清理
selectObject: "Strings fileList"
Remove

appendInfoLine: "--- 所有共振峰提取任务已完成 ---"