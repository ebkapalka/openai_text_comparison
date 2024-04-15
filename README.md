# OpenAI Text Comparison
### Eric Kapalka

This application was written to see if a GPT model cold do a batter job than fuzzy string matching when comparing text.  The sample data are two lists of college majors with slightly (or very) different naming conventions.  cappex_data is the major descriptions from CAPPEX, and banner_data.csv is the major descriptions from the Banner system.  This application performs the same comparisons using all 3 of OpenAI's Embeddings models and can be easily adapted if they ever add more (see https://platform.openai.com/docs/guides/embeddings/embedding-models)

This will require available credit on your OpenAI account (https://platform.openai.com/account/billing/overview), however Embeddings are the least expensive models by far.  $5 USD (at the time of writing) will allow for somewhere between 12,000,000 and 83,000,000 comparisons assuming an average of 3 tokens per input.  College major descriptions are small, typically 1-6 tokens.  With clever batching (which is supported), that can stretch even further.  Some theory and explanation:

&nbsp;

> ## What are Embeddings?
> Embeddings are a way of transforming objects like text, images, or even sounds into a numerical form (usually vectors of numbers) that a computer can understand and process. In the context of text embeddings, each piece of text (whether it's a word, sentence, or document) is converted into a vector in a high-dimensional space. The key idea behind embeddings is that they capture semantic meaning—texts that are similar in meaning are close to each other in this high-dimensional space, while dissimilar texts are far apart.
> 
> These vectors might look like nonsense numbers at first glance, but each dimension represents a latent feature of the text, learned from massive amounts of data. These features might capture aspects of the text's meaning, like topic, sentiment, or usage in a certain context.

> ## How Embeddings Compare Text Similarity
> Embeddings allow us to compute the similarity between texts quantitatively. After transforming texts into their vector forms, we can measure how close or far apart they are in the embedding space. This is where cosine similarity comes in.

> ## Cosine Similarity
> Cosine similarity is a metric used to measure how similar two vectors are irrespective of their magnitude. Mathematically, it calculates the cosine of the angle between two vectors projected in a multi-dimensional space.  This is how we calculate the similarity between two texts using embeddings. The cosine similarity score ranges from -1 to 1, where 1 means the texts are identical, 0 means they are orthogonal (i.e., unrelated), and -1 means they are diametrically opposed.
