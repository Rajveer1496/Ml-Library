package gui;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

public class TrainHandler implements HttpHandler {
    private final BackendService backendService = new BackendService();

    @Override
    public void handle(HttpExchange exchange) throws IOException {
        if ("POST".equalsIgnoreCase(exchange.getRequestMethod())) {
            try (InputStream is = exchange.getRequestBody()) {
                // Read the JSON request from the frontend
                String requestBody = new String(is.readAllBytes(), StandardCharsets.UTF_8);

                // Process the request using the existing backend service
                String response = backendService.handleTrainRequest(requestBody);

                // Send the JSON response back to the frontend
                byte[] responseBytes = response.getBytes(StandardCharsets.UTF_8);
                exchange.getResponseHeaders().set("Content-Type", "application/json");
                exchange.sendResponseHeaders(200, responseBytes.length);
                try (OutputStream os = exchange.getResponseBody()) {
                    os.write(responseBytes);
                }
            } catch (Exception e) {
                e.printStackTrace();
                String errorResponse = "{\"status\": \"error\", \"message\": \"Internal server error.\"}";
                byte[] responseBytes = errorResponse.getBytes(StandardCharsets.UTF_8);
                exchange.getResponseHeaders().set("Content-Type", "application/json");
                exchange.sendResponseHeaders(500, responseBytes.length);
                try (OutputStream os = exchange.getResponseBody()) {
                    os.write(responseBytes);
                }
            }
        } else {
            // Only POST method is allowed
            exchange.sendResponseHeaders(405, -1); // 405 Method Not Allowed
        }
    }
}
