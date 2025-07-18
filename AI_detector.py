import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

class PerplexityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GPT-2 文章困惑度分析")
        self.root.geometry("800x650") # 設定視窗大小

        # 載入模型和分詞器 (在單獨的執行緒中載入，避免 GUI 凍結)
        self.model = None
        self.tokenizer = None
        self.model_name = "gpt2" # 可以換成 "gpt2-medium" 或 "gpt2-large"

        self._create_widgets()
        self._load_model_async()

    def _create_widgets(self):
        # 設定整體佈局的 padding
        main_frame = ttk.Frame(self.root, padding="15 15 15 15")
        main_frame.pack(fill="both", expand=True)
        main_frame.columnconfigure(0, weight=1)

        # 1. 輸入文本區域
        ttk.Label(main_frame, text="請輸入您的英文文章：", font=('Arial', 12, 'bold')).grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        self.input_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, width=80, height=15, font=('Arial', 10))
        self.input_text.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        # 綁定 ctrl+a 全選
        self.input_text.bind("<Control-a>", self.select_all_text)
        self.input_text.bind("<Command-a>", self.select_all_text) # For macOS

        # 2. 模型加載狀態
        self.status_var = tk.StringVar()
        self.status_var.set("正在載入模型，請稍候...")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var, font=('Arial', 10, 'italic'), foreground='blue')
        self.status_label.grid(row=2, column=0, sticky="w", pady=(0, 10))

        # 3. 計算按鈕
        self.calculate_button = ttk.Button(main_frame, text="計算困惑度", command=self._start_calculation, state=tk.DISABLED)
        self.calculate_button.grid(row=3, column=0, sticky="ew", pady=(0, 20))

        # 4. 結果顯示區域
        ttk.Label(main_frame, text="=== 分析結果 ===", font=('Arial', 12, 'bold')).grid(row=4, column=0, sticky="w", pady=(10, 5))

        # 困惑度
        self.ppl_var = tk.StringVar()
        ttk.Label(main_frame, text="整體平均困惑度（PPL）：", font=('Arial', 11)).grid(row=5, column=0, sticky="w", padx=(10, 0))
        ttk.Label(main_frame, textvariable=self.ppl_var, font=('Arial', 11, 'bold'), foreground='darkgreen').grid(row=5, column=0, sticky="e", padx=(0, 10))

        # Token 損失變異量
        self.var_loss_var = tk.StringVar()
        ttk.Label(main_frame, text="Token損失變異量：", font=('Arial', 11)).grid(row=6, column=0, sticky="w", padx=(10, 0))
        ttk.Label(main_frame, textvariable=self.var_loss_var, font=('Arial', 11, 'bold'), foreground='darkgreen').grid(row=6, column=0, sticky="e", padx=(0, 10))

        # 判斷結果
        self.prediction_var = tk.StringVar()
        ttk.Label(main_frame, text="判斷結果：", font=('Arial', 11)).grid(row=7, column=0, sticky="w", pady=(10, 5), padx=(10, 0))
        self.prediction_label = ttk.Label(main_frame, textvariable=self.prediction_var, font=('Arial', 12, 'bold'), foreground='purple')
        self.prediction_label.grid(row=7, column=0, sticky="e", pady=(10, 5), padx=(0, 10))

        # 底部說明
        ttk.Label(main_frame, text="提示：PPL 越低，通常表示文本對模型而言越容易預測。", font=('Arial', 9, 'italic')).grid(row=8, column=0, sticky="w", pady=(10, 0))
        ttk.Label(main_frame, text="注意：這些判斷閾值是經驗性的，可能需要根據實際應用調整。", font=('Arial', 9, 'italic')).grid(row=9, column=0, sticky="w", pady=(0, 5))

        # 設置行和列的權重，讓其隨視窗大小變化
        main_frame.rowconfigure(1, weight=1) # 輸入文本區域可以擴展

    def select_all_text(self, event=None):
        self.input_text.tag_add("sel", "1.0", "end-1c")
        return "break" # Prevents the default behavior of the event

    def _load_model_async(self):
        """在單獨的執行緒中加載模型，避免阻塞 GUI"""
        self.calculate_button.config(state=tk.DISABLED)
        self.status_var.set(f"正在載入 GPT-2 模型 ({self.model_name})... 請稍候...")
        self.root.update_idletasks() # 強制更新 GUI

        def load_task():
            try:
                self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
                self.tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
                self.model.eval()
                self.root.after(0, self._on_model_loaded, True) # 回到主執行緒更新 GUI
            except Exception as e:
                self.root.after(0, self._on_model_loaded, False, str(e))

        threading.Thread(target=load_task).start()

    def _on_model_loaded(self, success, error_message=None):
        if success:
            self.status_var.set(f"模型 {self.model_name} 載入完成，可以開始分析。")
            self.calculate_button.config(state=tk.NORMAL)
        else:
            self.status_var.set(f"模型載入失敗: {error_message}")
            messagebox.showerror("錯誤", f"無法載入模型：{error_message}\n請檢查網絡連接或模型名稱。")
            self.calculate_button.config(state=tk.DISABLED)

    def _start_calculation(self):
        """啟動計算，並在單獨的執行緒中執行"""
        text = self.input_text.get("1.0", "end-1c").strip() # 獲取文本並移除末尾換行符

        if not text:
            messagebox.showwarning("輸入錯誤", "請在文本框中輸入文章後再計算。")
            return

        if self.model is None or self.tokenizer is None:
            messagebox.showerror("錯誤", "模型尚未載入完成，請稍候。")
            return

        self.calculate_button.config(state=tk.DISABLED)
        self.status_var.set("正在計算中...請稍候...")
        self.ppl_var.set("")
        self.var_loss_var.set("")
        self.prediction_var.set("")
        self.root.update_idletasks() # 強制更新 GUI

        def calculation_task():
            try:
                avg_ppl, var_token_losses, prediction_text = self._calculate_perplexity(text)
                self.root.after(0, self._on_calculation_complete, avg_ppl, var_token_losses, prediction_text)
            except Exception as e:
                self.root.after(0, self._on_calculation_error, str(e))

        threading.Thread(target=calculation_task).start()

    def _on_calculation_complete(self, avg_ppl, var_token_losses, prediction_text):
        """計算完成後更新 GUI"""
        self.ppl_var.set(f"{avg_ppl:.2f}" if avg_ppl != float('inf') else "N/A (文本過短)")
        self.var_loss_var.set(f"{var_token_losses:.2f}" if var_token_losses != float('inf') else "N/A (文本過短)")
        self.prediction_var.set(prediction_text)
        self.status_var.set("計算完成。")
        self.calculate_button.config(state=tk.NORMAL)

        # 根據預測結果調整預測文字的顏色
        if "極高可能是AI生成內容" in prediction_text:
            self.prediction_label.config(foreground='red')
        elif "可能是AI生成" in prediction_text:
            self.prediction_label.config(foreground='orange')
        elif "較可能是人類撰寫" in prediction_text:
            self.prediction_label.config(foreground='darkblue')
        elif "極高可能是人類撰寫" in prediction_text:
            self.prediction_label.config(foreground='green')
        else:
            self.prediction_label.config(foreground='purple') # 默認顏色

    def _on_calculation_error(self, error_message):
        """計算出錯時更新 GUI"""
        self.status_var.set("計算失敗。")
        messagebox.showerror("錯誤", f"計算過程中發生錯誤：{error_message}")
        self.calculate_button.config(state=tk.NORMAL)


    def _calculate_perplexity(self, text):
        """
        核心困惑度計算邏輯，與原始程式碼相同
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]

        avg_ppl = float('inf')
        var_token_losses = float('inf')
        prediction_text = "無法判斷 (文本過短或錯誤)"

        if input_ids.shape[1] <= 1:
            prediction_text = "⚠️ 警告：輸入文本過短，無法計算有效的困惑度。"
        else:
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                overall_loss = outputs.loss.item()
                logits = outputs.logits

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                token_losses_np = token_losses.detach().cpu().numpy()

                avg_ppl = np.exp(overall_loss)
                var_token_losses = np.var(token_losses_np)

            # 簡單判斷（可微調閾值）
            ai_ai_threshold = 30       # 如果 PPL 低於此值，可能傾向於 AI
            ai_mix_threshold = 100       # 如果 PPL 低於此值，可能傾向於 AI
            ai_var_loss_threshold = 13 # 如果損失變異量低於此值，可能傾向於 AI

            if avg_ppl < ai_ai_threshold and var_token_losses < ai_var_loss_threshold:
                prediction_text = "🤖 極高可能是AI生成內容 (PPL極低，高度可預測且平滑)"
            elif avg_ppl < ai_ai_threshold and var_token_losses >= ai_var_loss_threshold:
                prediction_text = "🤖 可能是AI生成，但包含非典型模式 (PPL低，但詞語預測難度波動較大)"
            elif avg_ppl >= ai_ai_threshold and avg_ppl < ai_mix_threshold and var_token_losses < ai_var_loss_threshold:
                prediction_text = "🤔 可能是AI生成或經過高度潤飾的內容 (PPL中等，但結構極為平穩)"
            elif avg_ppl >= ai_ai_threshold and avg_ppl < ai_mix_threshold and var_token_losses >= ai_var_loss_threshold:
                prediction_text = "✅ 較可能是人類撰寫 (PPL中等，語氣或表達具備自然波動)"
            else: # avg_ppl >= ai_mix_threshold
                prediction_text = "✅ 極高可能是人類撰寫 (PPL高，模型預測困難，符合人類寫作特點)"
        
        return avg_ppl, var_token_losses, prediction_text

if __name__ == "__main__":
    root = tk.Tk()
    app = PerplexityApp(root)
    root.mainloop()