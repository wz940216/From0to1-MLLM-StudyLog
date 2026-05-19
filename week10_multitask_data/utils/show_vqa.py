import json

ann_path = "dataset/VQA/abstract_v002_val2017_annotations.json"
ques_path = "dataset/VQA/OpenEnded_abstract_v002_val2017_questions.json"

with open(ann_path, "r") as f:
    anns = json.load(f)["annotations"]

with open(ques_path, "r") as f:
    ques = json.load(f)["questions"]

qid = 289402

ann = next(x for x in anns if x["question_id"] == qid)
q = next(x for x in ques if x["question_id"] == qid)

print("image_id:", ann["image_id"])
print("question:", q["question"])
print("answer:", ann["multiple_choice_answer"])
print("all answers:", [a["answer"] for a in ann["answers"]])