from datasets import load_dataset
import dspy.evaluate
from human_eval.data import read_problems
from human_eval_execute import check_correctness, unsafe_execute_tests
import os

os.environ["DSP_NOTEBOOK_CACHEDIR"] = "./human_eval_dspy_cache"

import dspy
from dspy.evaluate import Evaluate

turbo = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=1000)
dspy.settings.configure(lm=turbo)
dspy.configure(trace=[])


problems = read_problems()
human_eval_dataset = [
    dspy.Example(**x).with_inputs("prompt") for x in problems.values()
]


class CodeProblem(dspy.Signature):
    prompt = dspy.InputField(format=str)
    code = dspy.OutputField()


def post_process_code(code):
    code = code.split("</code>")[0]
    code = code.replace("```python", "")
    code = code.split("```")[0]
    code = code.replace("<code>", "")
    return code


def post_process_tests(tests):
    result = []
    for l in tests.split("\n"):
        if l.strip().startswith("assert"):
            assert_test = l.strip()
            eqns = assert_test[7:].split("==")
            if len(eqns) != 2:
                continue
            actual, expected = eqns
            assert_message = f"Expected {expected.strip()}."
            actual_result = "result = " + actual.strip() + "\n"
            result.append(f"{actual_result}\n{assert_test}, {repr(assert_message)}")
    return result
        

def human_eval_evaluate(example: dspy.Example, pred: dspy.Prediction):
    result = check_correctness(example.toDict(), post_process_code(pred.code), 2)
    return result["passed"]


class GenerateTests(dspy.Signature):
    prompt = dspy.InputField(format=str)
    tests = dspy.OutputField(
        desc="Executable tests using assert, you can use the one in prompts. \
                             Do not put tests in another local function, directly write them."
    )


def generate_tests(prompt):
    test_gen = dspy.ChainOfThought(GenerateTests)
    tests = test_gen(prompt=prompt)
    return tests.tests


def generate_and_check(prompt, pred, task_id=0):
    code = post_process_code(pred.code)
    tests = post_process_tests(post_process_code(generate_tests(prompt)))

    if len(tests) == 0:
        return True, "No tests found", "No tests found"

    for test in tests:
        result = check_correctness(
            {"prompt": prompt, "entry_point": "dummy", "test": test, "task_id": task_id},
            code,
            2,
            eval_fun=unsafe_execute_tests
        )
        if not result["passed"]:
            break
    return result["passed"], test, result["result"]


class CodeGenerator(dspy.Module):
    def __init__(self):
        self.prog = dspy.ChainOfThought("prompt -> code")

    def forward(self, prompt):
        pred = self.prog(prompt=prompt)
        passed, test, error = generate_and_check(prompt, pred)
        dspy.Suggest(
            passed, f"The generated code failed the test {test}, please fix the error: {error}.", target_module=self.prog
        )
        return pred

class NaiveCodeGenerator(dspy.Module):
    def __init__(self):
        self.prog = dspy.ChainOfThought("prompt -> code")

    def forward(self, prompt):
        pred = self.prog(prompt=prompt)
        return pred


if __name__ == "__main__":
    evaluator = Evaluate(
        devset=human_eval_dataset,
        metric=human_eval_evaluate,
        num_threads=1, # DO NOT CHANGE, otherwise OpenAI evaluate will complain
        display_progress=True,
    )
    program = CodeGenerator().activate_assertions()
    naive_program = NaiveCodeGenerator()
    print("Evaluating CodeGenerator with Suggestions:")
    evaluator(program=program)
    print("Evaluating NaiveCodeGenerator:")
    evaluator(program=naive_program)