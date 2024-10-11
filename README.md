## Abilities Tested
1. **Arithmetic**: capability to perform mathematical calculations
2. **Logic**: ability to apply logical operations and make decisions based on input
3. **Long-Range**: capacity to handle and process long sequences of input effectively
3. **Inner-State**: ability to maintain and manipulate internal representations of information during processing
5. **Memory**: ability to recall exact information from training data
6. **Size/Efficiency**: (FLOPs + Params) required for inference

## Input Format
- No tokenizer
- Input will be just characters, mostly digits and special symbols

## Tasks
Smallest amount of tasks to test on that can rate a model on the radar chart:

1. Evaluate long mathematical expressions (Arithmetic, Logic, Long-Range)
2. Match opening and closing brackets, output if balanced (Long-Range)
3. Recall long random strings from the training data (Memory, Long-Range)
4. Apply ops to state `321_3:SwapLeftIncrementIncrementSwap -> 5213_` (Logic, Inner-State)
