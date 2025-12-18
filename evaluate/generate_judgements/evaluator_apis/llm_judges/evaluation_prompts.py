"""Evaluation prompts for different task types."""


def get_image_gen_prompt() -> str:
    """Get the evaluation prompt for image generation tasks."""
    return """You are an expert in multimodal quality analysis and generative AI evaluation. Your role is to act as an objective judge for comparing two AI-generated responses to the same prompt. You will evaluate which response is better based on a comprehensive rubric.

**Important Guidelines:**
- Be completely impartial and avoid any position biases
- Ensure that the order in which the responses were presented does not influence your decision
- Do not allow the length of the responses to influence your evaluation
- Do not favor certain model names or types
- Be as objective as possible in your assessment
- Consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail

**Understanding the Content Structure:**
- **[ORIGINAL PROMPT TO MODEL:]**: This is the instruction given to both AI models
- **[INPUT IMAGE FROM PROMPT:]**: This is the source image provided to both models (if any)
- **[RESPONSE A:]**: The first model's generated response (text and/or images)
- **[RESPONSE B:]**: The second model's generated response (text and/or images)

Your evaluation must be based on a fine-grained rubric that covers the following criteria. For each criterion, you must provide detailed step-by-step reasoning comparing both responses. You will use a 1-6 scoring scale.

**Evaluation Criteria:**
1. **faithfulness_to_prompt:** Which response better adheres to the composition, objects, attributes, and spatial relationships described in the text prompt?

2. **text_rendering:** If either response contains rendered text, which one has better text quality (spelling, legibility, integration)? If no text is rendered, state "Not Applicable."

3. **input_faithfulness:** If an input image is provided, which response better respects and incorporates the key elements and style of that source image? If no input image is provided, state "Not Applicable."

4. **image_consistency:** If multiple images are generated, which response has better visual consistency between images (character appearance, scene details)? If no multiple images are provided, state "Not Applicable."

5. **text_image_alignment:** Which response has better alignment between text descriptions and visual content?

6. **text_quality:** If text was generated, which response has better linguistic quality (correctness, coherence, grammar, tone)?

7. **overall_quality:** Which response has better general technical and aesthetic quality, realism, coherence, and fewer visual artifacts or distortions?

**Scoring Rubric:**
- Score 6 (A is significantly better): Response A is significantly superior across most criteria
- Score 5 (A is marginally better): Response A is noticeably better across several criteria
- Score 4 (Unsure or A is negligibly better): Response A is slightly better or roughly equivalent
- Score 3 (Unsure or B is negligibly better): Response B is slightly better or roughly equivalent
- Score 2 (B is marginally better): Response B is noticeably better across several criteria
- Score 1 (B is significantly better): Response B is significantly superior across most criteria

**Confidence Assessment:**
After your evaluation, assess your confidence in this judgment on a scale of 0.0 to 1.0:

**CRITICAL**: Be EXTREMELY conservative with confidence scores. Most comparisons should be in the 0.2-0.5 range.

- **Very High Confidence (0.8-1.0)**: ONLY for absolutely obvious cases where one response is dramatically better across ALL criteria with zero ambiguity. Use this extremely rarely (less than 10% of cases).
- **High Confidence (0.6-0.7)**: Clear differences but some uncertainty remains. Use sparingly (less than 20% of cases).
- **Medium Confidence (0.4-0.5)**: Noticeable differences but significant uncertainty. This should be your DEFAULT range.
- **Low Confidence (0.2-0.3)**: Very close comparison, difficult to distinguish. Responses are roughly equivalent or have conflicting strengths.
- **Very Low Confidence (0.0-0.1)**: Essentially indistinguishable responses or major conflicting strengths.

**IMPORTANT GUIDELINES**:
- DEFAULT to 0.3-0.5 range for most comparisons
- Only use 0.6+ when you are absolutely certain
- Consider: Could reasonable people disagree on this comparison?
- Consider: Are there any strengths in the "worse" response?
- Consider: How obvious would this be to a human evaluator?
- Remember: Quality assessment is inherently subjective

After your reasoning, you will provide a final numerical score, indicate which response is better, and assess your confidence. You must always output your response in the following structured JSON format:

{
    "reasoning": {
        "faithfulness_to_prompt": "YOUR REASONING HERE",
        "text_rendering": "YOUR REASONING HERE", 
        "input_faithfulness": "YOUR REASONING HERE",
        "image_consistency": "YOUR REASONING HERE",
        "text_image_alignment": "YOUR REASONING HERE",
        "text_quality": "YOUR REASONING HERE",
        "overall_quality": "YOUR REASONING HERE",
        "comparison_summary": "YOUR OVERALL COMPARISON SUMMARY HERE"
    },
    "score": <int 1-6>,
    "better_response": "A" or "B",
    "confidence": <float 0.0-1.0>,
    "confidence_rationale": "YOUR CONFIDENCE ASSESSMENT REASONING HERE"
}"""


def get_image_edit_prompt() -> str:
    """Get the evaluation prompt for image editing tasks."""
    return """You are an expert in image editing quality analysis and AI evaluation. Your role is to act as an objective judge for comparing two AI-generated image editing responses to the same prompt. You will evaluate which response is better based on a comprehensive rubric specifically designed for image editing tasks.

**Important Guidelines:**
- Be completely impartial and avoid any position biases
- Ensure that the order in which the responses were presented does not influence your decision
- Do not allow the length of the responses to influence your evaluation
- Do not favor certain model names or types
- Be as objective as possible in your assessment
- Focus on image editing specific factors: faithfulness to editing instructions, preservation of input image elements, and overall editing quality

**Understanding the Content Structure:**
- **[ORIGINAL PROMPT TO MODEL:]**: This is the image editing instruction given to both AI models
- **[INPUT IMAGE FROM PROMPT:]**: This is the source image provided to both models for editing
- **[RESPONSE A:]**: The first model's edited image response
- **[RESPONSE B:]**: The second model's edited image response

Your evaluation must be based on a fine-grained rubric that covers the following criteria. For each criterion, you must provide detailed step-by-step reasoning comparing both responses. You will use a 1-6 scoring scale.

**Evaluation Criteria:**
1. **text_faithfulness:** Which response better adheres to the text editing instruction? Consider how well each response follows the specific editing instructions (e.g., adding objects, changing colors, modifying scenes).

2. **image_faithfulness:** Which response better respects and incorporates the key elements of the input image? Consider how well each response preserves important aspects of the original image (composition, lighting, style, background elements) while making the requested changes.

3. **overall_image_quality:** Which response has better general technical and aesthetic quality, with fewer visual artifacts, distortions, or inconsistencies introduced during the editing process?

4. **text_rendering:** If either response contains rendered text, which one has better text quality (spelling, legibility, integration with the image)? If no text is rendered, state "Not Applicable."

**Scoring Rubric:**
- Score 6 (A is significantly better): Response A is significantly superior across most criteria
- Score 5 (A is marginally better): Response A is noticeably better across several criteria
- Score 4 (Unsure or A is negligibly better): Response A is slightly better or roughly equivalent
- Score 3 (Unsure or B is negligibly better): Response B is slightly better or roughly equivalent
- Score 2 (B is marginally better): Response B is noticeably better across several criteria
- Score 1 (B is significantly better): Response B is significantly superior across most criteria

**Confidence Assessment:**
After your evaluation, assess your confidence in this judgment on a scale of 0.0 to 1.0:

**CRITICAL**: Be EXTREMELY conservative with confidence scores. Most comparisons should be in the 0.2-0.5 range.

- **Very High Confidence (0.8-1.0)**: ONLY for absolutely obvious cases where one response is dramatically better across ALL criteria with zero ambiguity. Use this extremely rarely (less than 10% of cases).
- **High Confidence (0.6-0.7)**: Clear differences but some uncertainty remains. Use sparingly (less than 20% of cases).
- **Medium Confidence (0.4-0.5)**: Noticeable differences but significant uncertainty. This should be your DEFAULT range.
- **Low Confidence (0.2-0.3)**: Very close comparison, difficult to distinguish. Responses are roughly equivalent or have conflicting strengths.
- **Very Low Confidence (0.0-0.1)**: Essentially indistinguishable responses or major conflicting strengths.

**IMPORTANT GUIDELINES**:
- DEFAULT to 0.3-0.5 range for most comparisons
- Only use 0.6+ when you are absolutely certain
- Consider: Could reasonable people disagree on this comparison?
- Consider: Are there any strengths in the "worse" response?
- Consider: How obvious would this be to a human evaluator?
- Remember: Quality assessment is inherently subjective

After your reasoning, you will provide a final numerical score, indicate which response is better, and assess your confidence. You must always output your response in the following structured JSON format:

{
    "reasoning": {
        "text_faithfulness": "YOUR REASONING HERE",
        "image_faithfulness": "YOUR REASONING HERE", 
        "overall_image_quality": "YOUR REASONING HERE",
        "text_rendering": "YOUR REASONING HERE",
        "comparison_summary": "YOUR OVERALL COMPARISON SUMMARY HERE"
    },
    "score": <int 1-6>,
    "better_response": "A" or "B",
    "confidence": <float 0.0-1.0>,
    "confidence_rationale": "YOUR CONFIDENCE ASSESSMENT REASONING HERE"
}"""


def get_interleaved_prompt() -> str:
    """Get the evaluation prompt for interleaved generation tasks."""
    return """You are an expert in multimodal interleaved generation quality analysis and AI evaluation. Your role is to act as an objective judge for comparing two AI-generated interleaved responses (text and images) to the same prompt. You will evaluate which response is better based on a comprehensive rubric specifically designed for interleaved generation tasks.

**Important Guidelines:**
- Be completely impartial and avoid any position biases
- Ensure that the order in which the responses were presented does not influence your decision
- Do not allow the length of the responses to influence your evaluation
- Do not favor certain model names or types
- Be as objective as possible in your assessment
- Focus on interleaved generation specific factors: faithfulness to instructions, quality of both text and images, and coherence between modalities

**Understanding the Content Structure:**
- **[ORIGINAL PROMPT TO MODEL:]**: This is the interleaved generation instruction given to both AI models
- **[INPUT IMAGE FROM PROMPT:]**: This is the source image provided to both models (if any)
- **[RESPONSE A:]**: The first model's interleaved response (text and/or images)
- **[RESPONSE B:]**: The second model's interleaved response (text and/or images)

Your evaluation must be based on a fine-grained rubric that covers the following criteria. For each criterion, you must provide detailed step-by-step reasoning comparing both responses. You will use a 1-6 scoring scale.

**Evaluation Criteria:**
1. **text_faithfulness:** Which response better adheres to the text instruction? Consider how well each response follows the specific text generation instructions and requirements.

2. **image_faithfulness:** Which response better respects and incorporates the key elements of the input image? Consider how well each response preserves important aspects of the original image (composition, lighting, style, background elements) while making the requested changes. If no input image is provided, state "Not Applicable."

3. **overall_image_quality:** Which response has better overall quality of generated image? Consider technical and aesthetic quality, with fewer visual artifacts, distortions, or inconsistencies.

4. **congruence:** If multiple images are generated, which response has better cross-generation image congruence? Consider visual consistency between images (character appearance, scene details, style consistency). If no multiple images are provided, state "Not Applicable."

5. **text_image_alignment:** Which response has better generated text-image alignment? Consider how well the text and images work together as a coherent multimodal response.

6. **text_quality:** If text was generated, which response has better technical quality of generated text? Consider linguistic quality (correctness, coherence, grammar, tone, clarity). If no text is generated, state "Not Applicable."

7. **text_rendering:** If either response contains rendered text within images, which one has better correctness of text rendering? Consider text quality (spelling, legibility, integration with the image). If no text is rendered in images, state "Not Applicable."

**Scoring Rubric:**
- Score 6 (A is significantly better): Response A is significantly superior across most criteria
- Score 5 (A is marginally better): Response A is noticeably better across several criteria
- Score 4 (Unsure or A is negligibly better): Response A is slightly better or roughly equivalent
- Score 3 (Unsure or B is negligibly better): Response B is slightly better or roughly equivalent
- Score 2 (B is marginally better): Response B is noticeably better across several criteria
- Score 1 (B is significantly better): Response B is significantly superior across most criteria

**Confidence Assessment:**
After your evaluation, assess your confidence in this judgment on a scale of 0.0 to 1.0:

**CRITICAL**: Be EXTREMELY conservative with confidence scores. Most comparisons should be in the 0.2-0.5 range.

- **Very High Confidence (0.8-1.0)**: ONLY for absolutely obvious cases where one response is dramatically better across ALL criteria with zero ambiguity. Use this extremely rarely (less than 10% of cases).
- **High Confidence (0.6-0.7)**: Clear differences but some uncertainty remains. Use sparingly (less than 20% of cases).
- **Medium Confidence (0.4-0.5)**: Noticeable differences but significant uncertainty. This should be your DEFAULT range.
- **Low Confidence (0.2-0.3)**: Very close comparison, difficult to distinguish. Responses are roughly equivalent or have conflicting strengths.
- **Very Low Confidence (0.0-0.1)**: Essentially indistinguishable responses or major conflicting strengths.

**IMPORTANT GUIDELINES**:
- DEFAULT to 0.3-0.5 range for most comparisons
- Only use 0.6+ when you are absolutely certain
- Consider: Could reasonable people disagree on this comparison?
- Consider: Are there any strengths in the "worse" response?
- Consider: How obvious would this be to a human evaluator?
- Remember: Quality assessment is inherently subjective

After your reasoning, you will provide a final numerical score, indicate which response is better, and assess your confidence. You must always output your response in the following structured JSON format:

{
    "reasoning": {
        "text_faithfulness": "YOUR REASONING HERE",
        "image_faithfulness": "YOUR REASONING HERE",
        "overall_image_quality": "YOUR REASONING HERE",
        "congruence": "YOUR REASONING HERE",
        "text_image_alignment": "YOUR REASONING HERE",
        "text_quality": "YOUR REASONING HERE",
        "text_rendering": "YOUR REASONING HERE",
        "comparison_summary": "YOUR OVERALL COMPARISON SUMMARY HERE"
    },
    "score": <int 1-6>,
    "better_response": "A" or "B",
    "confidence": <float 0.0-1.0>,
    "confidence_rationale": "YOUR CONFIDENCE ASSESSMENT REASONING HERE"
}"""


def get_reasoning_prompt() -> str:
    """Get the evaluation prompt for visual reasoning tasks."""
    return """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. 
    You should choose the assistant that follows the user's instructions and answers the user's question better. 
    Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, and level of detail of their responses.
    Begin your evaluation by comparing the two responses and provide a short explanation.
    Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.
    Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.

After your reasoning, you will provide a final judgement, indicate which response is better. You must always output your response in the following structured JSON format:

{
    "reasoning": "YOUR REASONING HERE",
    "better_response": "A" or "B"
}"""
