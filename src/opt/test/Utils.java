package opt.test;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

public class Utils {

	/** Write string to file at given path */
	public static synchronized void writeOutputToFile(String outputDir, String fileName, String string) {
		try {
			String full_path = outputDir + "/" + fileName;
			Path p = Paths.get(full_path);
			Files.createDirectories(p.getParent());
			Files.write(p, string.getBytes(), StandardOpenOption.CREATE, StandardOpenOption.APPEND);
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

}
