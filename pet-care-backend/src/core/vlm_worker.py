"""GenAI Worker - VLM + LLM 온디맨드 로딩/해제 (v3 신규 작성)"""
import torch
import gc
import json
import threading
import logging
import cv2
from PIL import Image

logger = logging.getLogger(__name__)


class GenAIWorker:
    """
    부정 감정 3초 지속 시 트리거.
    SmolVLM2-256M으로 장면 묘사 → Qwen2.5-3B로 행동지시 생성.
    각 모델은 순차 로딩하고 사용 후 즉시 메모리 해제 (6GB VRAM 제약).
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_running = False
        self.vlm_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
        self.llm_id = "Qwen/Qwen2.5-3B-Instruct"

    def run_async(self, frame_bgr, emotion: str, callback):
        """비동기 파이프라인 실행. callback(result_dict) 호출."""
        if self.is_running:
            logger.warning("GenAI Worker 이미 실행 중 — 요청 건너뜀")
            return
        self.is_running = True
        t = threading.Thread(
            target=self._run_pipeline, args=(frame_bgr, emotion, callback), daemon=True
        )
        t.start()

    def _release_gpu(self):
        """GPU 메모리 즉시 해제"""
        torch.cuda.empty_cache()
        gc.collect()

    def _run_pipeline(self, frame_bgr, emotion: str, callback):
        import json # (re 임포트 삭제)
        result = {"action": "처리중...", "control": None}
        vlm_desc = ""

        try:
            from transformers import (
                AutoProcessor,
                AutoModelForImageTextToText,
                AutoTokenizer,
                AutoModelForCausalLM,
                BitsAndBytesConfig,
            )
            import gc # 확실한 메모리 해제를 위해 임포트

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            # ── VLM 단계 ──
            logger.info(f"VLM 로딩: {self.vlm_id}")
            processor = AutoProcessor.from_pretrained(self.vlm_id, trust_remote_code=True)
            vlm_model = AutoModelForImageTextToText.from_pretrained(
                self.vlm_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            vlm_model.eval()
            
            pil_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "First, describe the cat or dog's body language and posture. Second, describe the surrounding environment and what the pet is reacting to (e.g., a mirror, another animal, an object). Ignore any overlaid text, watermarks, or TV subtitles."},
                    ],
                }
            ]
            text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=text_prompt, images=[pil_image], return_tensors="pt")
            
            processed_inputs = {}
            for k, v in inputs.items():
                if k == "pixel_values":
                    processed_inputs[k] = v.to(self.device, dtype=vlm_model.dtype)
                else:
                    processed_inputs[k] = v.to(self.device)

            logger.info("VLM 상황 묘사 생성 중...")
            gen_ids = vlm_model.generate(**processed_inputs, max_new_tokens=50)
            
            raw_vlm_desc = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
            
            if "Assistant:" in raw_vlm_desc:
                vlm_desc = raw_vlm_desc.split("Assistant:")[-1].strip()
            else:
                vlm_desc = raw_vlm_desc.strip()
                
            logger.info(f"VLM 출력: {vlm_desc}")

            # 🚨 [메모리 누수 해결 1] 사용한 모든 텐서 변수를 명시적으로 폭파시킴
            del vlm_model, processor, inputs, processed_inputs, gen_ids
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("VLM 메모리 해제 완료")

            # ── LLM 단계 ──
            logger.info(f"LLM 로딩: {self.llm_id}")
            tokenizer = AutoTokenizer.from_pretrained(self.llm_id)
            llm_model = AutoModelForCausalLM.from_pretrained(
                self.llm_id,
                quantization_config=bnb_config,
                device_map="auto",
            )

            llm_prompt = (
                f"Detected Emotion: {emotion}\n"
                f"Visual Context: {vlm_desc}\n\n"
                "Task: Provide a short guideline for the owner in Korean.\n"
                "Response Format: JSON only.\n"
                "Schema:\n"
                '{\n'
                '  "action": "Description for UI in Korean (한글)",\n'
                '  "control": {\n'
                '    "cmd": "MOVE_FORWARD|STOP|WAIT",\n'
                '    "param": "speed or duration"\n'
                '  }\n'
                '}\n'
                'Example: {"action": "고양이가 하악질을 하니 다가가지 마세요.", '
                '"control": {"cmd": "WAIT", "param": "2000"}}'
            )

            msgs = [{"role": "user", "content": llm_prompt}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(self.device)

            logger.info("행동 가이드 생성 중...")
            gen_ids = llm_model.generate(**model_inputs, max_new_tokens=150)
            generated_ids = [
                out[len(inp):] for inp, out in zip(model_inputs.input_ids, gen_ids)
            ]
            full_resp = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # 🚀 [파싱 오류 해결] 가장 첫 '{' 와 가장 마지막 '}' 를 찾아 완벽한 JSON 블록 추출
            start_idx = full_resp.find("{")
            end_idx = full_resp.rfind("}")
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = full_resp[start_idx : end_idx + 1]
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError:
                    result["action"] = "가이드 파싱 실패 (명령어 오류)"
            else:
                result["action"] = "가이드 생성 실패"

            # 🚨 [메모리 누수 해결 2] 사용한 모든 텐서 변수를 명시적으로 폭파시킴
            del llm_model, tokenizer, model_inputs, gen_ids, generated_ids
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("GenAI 파이프라인 완료. 메모리 해제됨.")

        except Exception as e:
            logger.error(f"GenAI 오류: {e}")
            result["action"] = "분석 중 시스템 오류 발생"

        finally:
            self.is_running = False
            if callback:
                callback(result)
