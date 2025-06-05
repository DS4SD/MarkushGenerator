import org.openscience.cdk.Atom;
import org.openscience.cdk.CDKConstants;
import org.openscience.cdk.interfaces.*;
import org.openscience.cdk.silent.SilentChemObjectBuilder;
import org.openscience.cdk.smiles.SmilesParser;
import org.openscience.cdk.smiles.smarts.SmartsPattern;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;
import org.openscience.cdk.smiles.*;
import org.openscience.cdk.depict.DepictionGenerator;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.exception.InvalidSmilesException;
import org.openscience.cdk.graph.Cycles;
import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.aromaticity.Aromaticity;
import org.openscience.cdk.aromaticity.ElectronDonation;
import org.openscience.cdk.isomorphism.Pattern;
import org.openscience.cdk.smiles.smarts.SMARTSQueryTool;
import org.openscience.cdk.renderer.RendererModel;
import org.openscience.cdk.renderer.SymbolVisibility;
import org.openscience.cdk.renderer.generators.BasicBondGenerator;
import org.openscience.cdk.renderer.generators.IGeneratorParameter;
import org.openscience.cdk.renderer.generators.standard.SelectionVisibility;
import org.openscience.cdk.renderer.generators.standard.StandardGenerator;
import org.openscience.cdk.renderer.color.UniColor;
import org.openscience.cdk.isomorphism.Pattern;
import org.openscience.cdk.renderer.RendererModel;
import org.openscience.cdk.renderer.AtomContainerRenderer;
import org.openscience.cdk.renderer.generators.BasicSceneGenerator;
import org.openscience.cdk.renderer.font.AWTFontManager;
import org.openscience.cdk.geometry.GeometryUtil;
import org.openscience.cdk.layout.StructureDiagramGenerator;
import org.openscience.cdk.depict.Abbreviations;
import org.openscience.cdk.io.MDLV3000Writer;
import org.openscience.cdk.depict.DepictionGenerator;
import org.openscience.cdk.renderer.generators.BasicBondGenerator;
import org.openscience.cdk.renderer.generators.IGenerator;
import org.openscience.cdk.renderer.generators.standard.StandardGenerator;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.tools.CDKHydrogenAdder;  
import java.io.FileWriter;
import javax.vecmath.Point2d;
import java.util.Random;
import java.util.Arrays;
import java.util.List;
import java.awt.Color;
import java.awt.Font;
import java.awt.FontFormatException;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;

//@SuppressWarnings("deprecation")
public class Depictor {
	public Depictor(String smiles, String index, String dataset) throws FontFormatException, IOException, CDKException {
		this.generate_image_masks_label(smiles, index,  dataset);
	}
	
	public void generate_image_masks_label(String smiles, String index, String  dataset) throws FontFormatException, IOException, CDKException{
	    String basePath = Paths.get(Depictor.class.getClassLoader().getResource("Depictor.class").getPath()).getParent().toString();
	    Random random = new Random();

		// Read SMILES
	    IChemObjectBuilder builder = SilentChemObjectBuilder.getInstance();
	    SmilesParser parser = new SmilesParser(builder);
	    IAtomContainer molecule = parser.parseSmiles(smiles);
	    // molecule = AtomContainerManipulator.removeHydrogens(molecule); // Remove explicit hydrogens
	    
        // Randomly make some hydrogens explicit (Currently not usable as for the OCR cells extraction, the molecule is read from the original CXSMILES, which is not updated.)
		// if (random.nextDouble() < 1) { //0.05) {
		// 	for (IAtom atom : molecule.atoms()) {
		// 		if (atom.getImplicitHydrogenCount() != null && atom.getImplicitHydrogenCount() > 0) {
		// 			if (random.nextDouble() < 1) { //0.15) {
		// 				int implicitHCount = atom.getImplicitHydrogenCount();
		// 				for (int i = 0; i < implicitHCount; i++) {
		// 					IAtom hydrogen = new Atom("H");
		// 					molecule.addAtom(hydrogen);
		// 					IBond bond = builder.newInstance(IBond.class, atom, hydrogen);
		// 					molecule.addBond(bond);
		// 				}
		// 				atom.setImplicitHydrogenCount(0);  
		// 			}
		// 		}
		// 	}
		// }

	    StructureDiagramGenerator sdg = new StructureDiagramGenerator();
	    sdg.setMolecule(molecule);
	    sdg.generateCoordinates(molecule);
	    molecule = sdg.getMolecule();	    

		// Select random drawing parameters
		double strokeRatio = 0.3 + (3.0 - 0.3)*random.nextDouble(); // Bond width
		double bondSeparation = 0.1 + (0.3 - 0.1)*random.nextDouble(); // Double bonds spacing
		double symbolMarginRatio = 0.5 + (5.5 - 0.5)*random.nextDouble(); // Atom label spacing

		boolean containsSubscript = false;
		String[] fontNames = new String[]{"lora", "arial", "cambria", "times"}; 
		for (char ch : "₀₁₂₃₄₅₆₇₈₉".toCharArray()) {
			if (smiles.contains(String.valueOf(ch))) {
				fontNames = new String[]{"cambria"};
				containsSubscript = true;
				break;  
			}
		}
		if (smiles.contains(String.valueOf("'"))) {
			if (containsSubscript) {
				System.out.println("Subscripts and quotes are not be compatible");
			}
			fontNames = new String[]{"lora", "arial"};
		}		
		String randomFontName = fontNames[random.nextInt(fontNames.length)]; // Font
		float randomFontSize = 8.0f + (20.0f - 8.0f) * random.nextFloat(); // Font size

		// Debug values
		// double strokeRatio = 1.5; 
		// double bondSeparation = 0.2; 
		// double symbolMarginRatio = 3.5; 
		// String randomFontName = "lora";
		// float randomFontSize = 14.0f; 
		
	    // Create image generator (Arial, Cambria, Lora, Times)
	    Font font = Font.createFont(Font.TRUETYPE_FONT, new File(basePath + "/../../data/fonts/" + randomFontName + ".ttf"))
				.deriveFont(randomFontSize);	

	    DepictionGenerator imageGenerator = new DepictionGenerator(font)
	    		// Size
	    		.withSize(289, 289)    
				.withFillToFit()
				// Color
				.withBackgroundColor(Color.WHITE)
				.withAtomColors(new UniColor(Color.BLACK))
				.withAnnotationColor(Color.BLACK);

		// Apply additional random drawing options
		if (random.nextDouble() < 0.01) {
			imageGenerator = imageGenerator.withAtomNumbers();
        } 
		if (random.nextDouble() < 0.1) {
			imageGenerator = imageGenerator.withCarbonSymbols();
        } 
		if (random.nextDouble() < 0.2) {
			imageGenerator = imageGenerator.withAromaticDisplay();
        } 
		if (random.nextDouble() < 0.025) {
			imageGenerator = imageGenerator.withDeuteriumSymbol(true);
        } 
		if (random.nextDouble() < 0.2) {
			imageGenerator = imageGenerator.withTerminalCarbons();
        } 

		// Set drawing parameters
		imageGenerator = imageGenerator.withParam(StandardGenerator.BondSeparation.class, bondSeparation);
		imageGenerator = imageGenerator.withParam(StandardGenerator.StrokeRatio.class, strokeRatio);
		imageGenerator = imageGenerator.withParam(StandardGenerator.SymbolMarginRatio.class, symbolMarginRatio);

		// Draw molecule and save MOL file
	    imageGenerator.depict(molecule).writeTo(basePath + "/../../data/dataset/" + dataset + "/images/" + index + ".svg");
	    MDLV3000Writer writer = new MDLV3000Writer(new FileWriter(new File(basePath + "/../../data/dataset/" + dataset + "/molfiles/" + index + ".mol")));
	    writer.write(molecule);
	    writer.close();
	}
	    	
	public static void main(String[] args) throws Exception {
		new Depictor(args[0], args[1], args[2]);
	}
}

// Archive: Tests to get the atom center position 
// Add landmark atoms
// Point2d center = GeometryUtil.get2DCenter(molecule);
// Point2d delta = new Point2d(1.0, 1.0);
// delta.add(center);
// Atom dummy1 = new Atom("*", delta);
// delta = new Point2d(-1.0, -1.0);
// delta.add(center);
// Atom dummy2 = new Atom("*", delta);
// molecule.addAtom(dummy1);
// molecule.addAtom(dummy2);
// GeometryUtil.rotate(molecule, new Point2d(0.0,0.0), 0.0);
// Point2d point = GeometryUtil.get2DCenter(molecule);
// boolean has2d = GeometryUtil.has2DCoordinates(molecule);
// for ( IAtom atom: molecule.atoms() ) {
// 	Point2d pt2 = atom.getPoint2d();
// 	System.out.println(pt2);
// }
// Abbreviations abrv = new Abbreviations();
// abrv.add("*C Me");
// abrv.add("*C(=O)OC CO2Me");
// abrv.add("*C(=O)O CO2H");
// int numAdded = abrv.apply(molecule);
// System.out.println(numAdded);