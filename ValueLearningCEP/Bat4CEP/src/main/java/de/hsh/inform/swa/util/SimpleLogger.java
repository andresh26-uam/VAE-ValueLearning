package de.hsh.inform.swa.util;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
/**
 * Basic logging class.
 * @author Software Architecture Research Group
 *
 */
public class SimpleLogger {
    private final List<PrintStream> streams = new ArrayList<>();

    public SimpleLogger(PrintStream out) {
        add(out);
    }

    public static SimpleLogger getSimpleLogger(File output) throws FileNotFoundException {
        SimpleLogger logger = new SimpleLogger(new PrintStream(output));
        return logger;
    }

    public SimpleLogger println(String text) {
        streams.forEach(s -> s.println(text));
        return this;
    }

    public SimpleLogger println() {
        streams.forEach(s -> s.println());
        return this;
    }

    public SimpleLogger add(PrintStream out) {
        streams.add(out);
        return this;
    }

    public SimpleLogger flush() {
        streams.forEach(s -> s.flush());
        return this;
    }

    public SimpleLogger println(Object obj) {
        streams.forEach(s -> s.println(obj));
        return this;

    }

    public void close() {
        streams.stream().filter(s -> s != System.out && s != System.err).forEach(s -> s.close());
    }

    public SimpleLogger print(String text) {
        streams.forEach(s -> s.print(text));
        return this;
    }

    public SimpleLogger print(int i) {
        streams.forEach(s -> s.print(i));
        return this;
    }

    public SimpleLogger print(char i) {
        streams.forEach(s -> s.print(i));
        return this;
    }

    public SimpleLogger print(long d) {
        streams.forEach(s -> s.print(d));
        return this;
    }
}
