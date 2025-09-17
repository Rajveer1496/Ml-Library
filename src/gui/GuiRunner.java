package gui;

import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.util.concurrent.Executors;

public class GuiRunner {
    public static void main(String[] args) throws IOException {
        int port = 8080;
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);

        // Handler for the API endpoint (/train)
        server.createContext("/train", new TrainHandler());

        // Handler for the API endpoint (/predict)
        server.createContext("/predict", new PredictHandler());

        // Handler for static files (/, /index.html, /script.js)
        // The base path is the 'gui' directory itself.
        server.createContext("/", new StaticFileHandler("src/gui"));

        // Use a fixed-size thread pool to handle requests concurrently
        server.setExecutor(Executors.newFixedThreadPool(10));

        // Start the server
        server.start();

        System.out.println("===================================================");
        System.out.println("ML Trainer Web Server has started!");
        System.out.println("Access the interface by opening your web browser to:");
        System.out.println("http://localhost:" + port);
        System.out.println("===================================================");
        System.out.println("Press Ctrl+C to stop the server.");
    }
}
