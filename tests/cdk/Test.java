import java.io.*;
import java.net.URL;
import java.nio.file.Paths;
import org.openscience.cdk.interfaces.IChemObjectBuilder;
import org.openscience.cdk.io.iterator.IteratingSDFReader;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.interfaces.IAtom;

public class Test {	
    """
    Test for the java and cdk installation.
    Run command: javac -cp "../../lib/*":. Test.java; java -cp "../../lib/*":. Test
    """
    public static void main(String[] args) {	
        FileReader sdfile = null;
        String basePath = Paths.get(Test.class.getClassLoader().getResource("Test.class").getPath()).getParent().toString();
        
        try {
            /* CDK does not automatically understand gzipped files */
            sdfile = new FileReader(new File(basePath + "/benzodiazepine.sdf"));
        } catch (FileNotFoundException e) {
            System.err.println("benzodiazepine.sdf not found");
            System.exit(1);
        }
        IteratingSDFReader mdliter = new IteratingSDFReader(sdfile,
                                            DefaultChemObjectBuilder.getInstance());
        while (mdliter.hasNext()) {
            IAtomContainer mol = (IAtomContainer) mdliter.next();
            int numHeavies = 0;
            for (IAtom atom : mol.atoms()) {
                if (atom.getAtomicNumber() > 1) {
                    numHeavies++;
                }
            }
            System.out.println(numHeavies);
        }
    }
}