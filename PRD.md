I need a very minimal simple playground to test llm request against csv dataset on predetermined prompt or sequence of prompts so i can observe the outputs and compare to the ones expected from the dataset

The journey would be as follow

1. Pick a task to experiment
Example: Message classification

2. Add prompt(s)
Example: 
Prompt A "You work for the following industry: {industry}. Summarize the main intent of this message into one sentence" 
Input A -> Industry
Input B -> User message
Output -> Summary
Prompt B "Classify the following summary of a message as GOOD or BAD"
Input -> Summary
Output -> Classification

3. Add CSV Dataset
Required columns:
- All Inputs needed for the prompt
- Expected final output
Example columns:
- A: Industry
- B: Message
- c: Expected Classification

4. Run an LLM request for each item on dataset with the prompt pre filled with the inputs

5. Add new columns in the dataset with each prompt output. If output is a json, each key should be a column. 