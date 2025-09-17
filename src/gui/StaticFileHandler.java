package gui;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.URI;

public class StaticFileHandler implements HttpHandler {
    private final String basePath;

    public StaticFileHandler(String basePath) {
        this.basePath = basePath;
    }

    @Override
    public void handle(HttpExchange exchange) throws IOException {
        URI uri = exchange.getRequestURI();
        String path = uri.getPath();
        if (path.equals("/")) {
            path = "/index.html"; // Serve index.html by default
        }

        File file = new File(basePath + path).getCanonicalFile();

        if (!file.getPath().startsWith(new File(basePath).getCanonicalPath())) {
            // Prevent directory traversal attacks
            sendError(exchange, 403, "Forbidden");
            return;
        }

        if (!file.exists() || file.isDirectory()) {
            sendError(exchange, 404, "Not Found");
            return;
        }

        // Set content type based on file extension
        String contentType = "application/octet-stream";
        if (path.endsWith(".html")) {
            contentType = "text/html";
        } else if (path.endsWith(".js")) {
            contentType = "application/javascript";
        }
        exchange.getResponseHeaders().set("Content-Type", contentType);

        // Stream the file content
        exchange.sendResponseHeaders(200, file.length());
        try (FileInputStream fs = new FileInputStream(file);
             OutputStream os = exchange.getResponseBody()) {
            fs.transferTo(os);
        }
    }

    private void sendError(HttpExchange exchange, int statusCode, String message) throws IOException {
        exchange.sendResponseHeaders(statusCode, message.length());
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(message.getBytes());
        }
    }
}
