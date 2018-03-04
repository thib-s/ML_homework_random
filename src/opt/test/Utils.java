package opt.test;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.text.SimpleDateFormat;
import java.util.Date;

public class Utils {

	public static synchronized void writeOutputToFile(String outputDir, String fileName, String string) {
		writeOutputToFile(outputDir, fileName, string, "");
	}

		/** Write string to file at given path */
	public static synchronized void writeOutputToFile(String outputDir, String fileName, String string, String header) {
		try {
			String full_path = outputDir + "/" + new SimpleDateFormat("dd-MM-yyyy").format(new Date()) + "_" + fileName;
			Path p = Paths.get(full_path);
			if (!Files.exists(p)) {
				string = header + string;
			}
			Files.createDirectories(p.getParent());
			Files.write(p, string.getBytes(), StandardOpenOption.CREATE, StandardOpenOption.APPEND);
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

}
