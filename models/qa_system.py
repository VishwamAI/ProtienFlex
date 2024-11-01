from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import logging

logger = logging.getLogger(__name__)

class ProteinQASystem:
    def __init__(self):
        try:
            model_name = "deepset/roberta-base-squad2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            self.model.eval()
            logger.info("QA system initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing QA system: {e}")
            raise

    def answer_question(self, context, question):
        try:
            inputs = self.tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)

            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1

            answer = self.tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end])
            confidence = float(torch.max(outputs.start_logits)) + float(torch.max(outputs.end_logits))

            return {
                "answer": answer,
                "confidence": confidence / 2
            }
        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
            return {
                "answer": "Unable to process question",
                "confidence": 0.0
            }
