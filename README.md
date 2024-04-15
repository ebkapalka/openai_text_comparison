# OpenAI Text Comparison
### Eric Kapalka

This application was written to see if a GPT model cold do a batter job than fuzzy string matching when comparing text.  The sample data are two lists of college majors with slightly (or very) different naming conventions.  cappex_data is the major descriptions from CAPPEX, and banner_data.csv is the major descriptions from the Banner system.  This application performs the same comparisons using all 3 of OpenAI's Embeddings models and can be easily adapted if they ever add more (see https://platform.openai.com/docs/guides/embeddings/embedding-models)

This will require available credit on your OpenAI account (https://platform.openai.com/account/billing/overview), however Embeddings are the least expensive models by far.  $5 USD (at the time of writing) will allow for somewhere between 12,000,000 and 83,000,000 comparisons assuming an average of 3 tokens per input.  College major descriptions are small, typically 1-6 tokens.
