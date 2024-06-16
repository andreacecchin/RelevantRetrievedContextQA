package com.andreacecchin;

// Configuration import - DO NOT DELETE
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.CosineSimilarity;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.time.Duration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import static java.util.stream.Collectors.joining;

// LLM import
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_3_5_TURBO;
import dev.langchain4j.model.vertexai.VertexAiChatModel;
import dev.langchain4j.model.vertexai.VertexAiGeminiChatModel;
import dev.langchain4j.model.anthropic.AnthropicChatModel;


public class Main {

    // * * * * * * * *
    // CONFIGURATION - DO NOT MODIFY
    public static String storePath = "src/main/resources/dataset/embedding.json";
    public static String datasetPath = "src/main/resources/dataset/dataset.json";
    public static String resultPath = "src/main/resources/result/result.json";  // Evaluation results will be stored here
    public static int maxResult = 5;
    public static double minScore = 0.65;
    // * * * * * * * *

    static class Benchmarking {

        public static void main(String[] args) {

            // * * * * * * * *
            // OFFICIAL RetrievedRelevantContextQA BENCHMARK CONFIGURATION FOR EMBEDDING STORE, EMBEDDING MODEL AND TEMPLATE
            // DO NOT MODIFY
            InMemoryEmbeddingStore<TextSegment> embeddingStore = InMemoryEmbeddingStore.fromFile(storePath);
            EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
            PromptTemplate promptTemplate = PromptTemplate.from(
                    """
                            Answer to the following question, based ONLY on the context i'll give you.

                            Question:---
                            {{question}}
                            ---

                            Context:---
                            {{information}}
                            --- end context ---

                            IF you have no useful information from context, answer with 'I can't provide any answer.'.
                            Don't use any other general knowledge to give information outside the context.
                            """);
            // * * * * * * * *



            // * * * * * * * *
            // CONFIGURATION FOR LLM
            // MODIFY WITH THE MODEL (SUPPORTED BY LANGCHAIN4J) YOU WANT TO EVALUATE
            // YOU MUST USE THE ChatLanguageModel CLASS
            // E.G.: OpenAI's gpt-3.5-turbo
            // Other examples at the end of this file
            ChatLanguageModel chatModel = OpenAiChatModel.builder()
                    .apiKey("demo")
                    .modelName(GPT_3_5_TURBO)
                    .temperature(0.4)
                    .build();
            // * * * * * * * *



            Gson gson = new GsonBuilder().setPrettyPrinting().create();

            try (FileReader reader = new FileReader(datasetPath)) {
                Type documentsType = new TypeToken<Map<String, List<Map<String, Object>>>>() {}.getType();
                Map<String, List<Map<String, Object>>> documentsMap = gson.fromJson(reader, documentsType);

                List<Map<String, Object>> documents = documentsMap.get("documents");
                for (Map<String, Object> document : documents) {
                    int documentId = ((Double) document.get("id")).intValue();
                    List<Map<String, Object>> questions = (List<Map<String, Object>>) document.get("questions");

                    for (Map<String, Object> question : questions) {
                        int questionId = ((Double) question.get("id")).intValue();
                        String actualQuestion = (String) question.get("actualQuestion");
                        String expectedAnswer = (String) question.get("expectedAnswer");
                        Map<String, Object> answer = (Map<String, Object>) question.get("answer");

                        // Question is embedded to search relevant information
                        Embedding questionEmbedding = embeddingModel.embed(actualQuestion).content();
                        List<EmbeddingMatch<TextSegment>> relevantEmbeddings
                                = embeddingStore.findRelevant(questionEmbedding, maxResult, minScore);

                        // Generation of final prompt
                        String information = relevantEmbeddings.stream()
                                .map(match -> match.embedded().text())
                                .collect(joining("\n--\n"));
                        Map<String, Object> variables = new HashMap<>();
                        variables.put("question", actualQuestion);
                        variables.put("information", information);
                        Prompt prompt = promptTemplate.apply(variables);

                        // Call to LLM, answer generation measuring time
                        long startTime = System.currentTimeMillis();
                        AiMessage aiMessage = chatModel.generate(prompt.toUserMessage()).content();
                        long endTime = System.currentTimeMillis();
                        long elapsedTime = endTime - startTime;
                        double finalTime = ((double) elapsedTime) / 1000;
                        String actualAnswer = aiMessage.text();

                        // Similarity between expected and actual answer with cosine similarity
                        Embedding expectedEmbedding = embeddingModel.embed(expectedAnswer).content();
                        Embedding actualEmbedding = embeddingModel.embed(actualAnswer).content();
                        double similarity = CosineSimilarity.between(expectedEmbedding, actualEmbedding);


                        // Write the result on the JSON
                        answer.put("actualAnswer", actualAnswer);  // answer generated by llm
                        if (similarity < 0.70) {
                            answer.put("correct", false);  // --> necessary human evaluation (modify the result.json)
                            if (expectedAnswer.equals("I can't provide any answer.")) {
                                answer.put("prompt", false);  // --> necessary human evaluation (modify the result.json)
                            } else {
                                answer.put("prompt", true);  // --> necessary human evaluation (modify the result.json)
                            }
                        } else {
                            answer.put("correct", true);  // --> necessary human evaluation (modify the result.json)
                            answer.put("prompt", true);  // --> necessary human evaluation (modify the result.json)
                        }
                        answer.put("hallucination", false);  // --> necessary human evaluation (modify the result.json)
                        answer.put("time", finalTime);  // time for answer generation
                        answer.put("similarity", similarity);  // semantic similarity (cosine similarity) between actual and expected answers

                        // Write on console the progression of the evaluation
                        System.out.println("Progress: " + (((documentId - 1) * 10) + questionId) + "%");

                        // Save the results after each modification
                        try (FileWriter writer = new FileWriter(resultPath)) {
                            gson.toJson(documentsMap, writer);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }

        }
    }

}

/* * * * *
OTHER POSSIBLE MODEL TO EVALUATE
Results of these models in this Benchmark can be found in generated_results directory


ChatLanguageModel chatModel = VertexAiChatModel.builder()
                    .project("gemini-synclab-proj")
                    .location("us-central1")
                    .modelName("chat-bison-32k")
                    .publisher("google")
                    .endpoint("us-central1-aiplatform.googleapis.com:443")
                    .temperature(0.4)
                    .build();


ChatLanguageModel chatModel = VertexAiGeminiChatModel.builder()
                    .project("gemini-synclab-proj")
                    .location("us-central1")
                    .modelName("gemini-1.5-pro-preview-0514")
                    .temperature(0.4F)
                    .build();


ChatLanguageModel chatModel = OpenAiChatModel.builder()
                    .apiKey("demo")
                    .modelName(GPT_3_5_TURBO)
                    .temperature(0.4)
                    .build();


ChatLanguageModel chatModel = OllamaChatModel.builder()
                    .baseUrl("http://localhost:11434")
                    .modelName("phi3")
                    .temperature(0.4)
                    .timeout(Duration.ofMinutes(5))
                    .build();


 * * * * */