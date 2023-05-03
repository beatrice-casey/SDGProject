import com.ibm.wala.cfg.ControlFlowGraph;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.*;
import com.ibm.wala.ipa.callgraph.impl.Everywhere;
import com.ibm.wala.ipa.callgraph.impl.Util;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.SSAPropagationCallGraphBuilder;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ipa.cha.ClassHierarchyFactory;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.ipa.slicer.*;
import com.ibm.wala.ssa.*;
import com.ibm.wala.types.ClassLoaderReference;
import com.ibm.wala.util.CancelException;
import com.ibm.wala.util.config.AnalysisScopeReader;
import com.ibm.wala.util.io.FileProvider;
import java.io.FileWriter;
import java.io.IOException;


import java.io.File;
import java.net.URL;
import java.util.ArrayList;
import java.util.Iterator;

import static java.lang.String.format;

public class Representations {

    public static void main(String[] args) throws CancelException, ClassHierarchyException, IOException {
        File exFile = new FileProvider().getFile("Java60RegressionExclusions.txt");

        for(int i = 0; i < 664; i++) {
            URL resource = getFilesForResource(i);
            AnalysisScope scope = AnalysisScopeReader.makeJavaBinaryAnalysisScope(resource.getPath(), exFile);
            String runtimeClasses = Representations.class.getResource("jdk-17.0.1/rt.jar").getPath();
            AnalysisScopeReader.addClassPathToScope(runtimeClasses, scope, ClassLoaderReference.Primordial);

            IClassHierarchy classHierarchy = ClassHierarchyFactory.make(scope);

            AnalysisOptions options = new AnalysisOptions();
            options.setEntrypoints(Util.makeMainEntrypoints(scope, classHierarchy));
            SSAPropagationCallGraphBuilder builder = Util.makeNCFABuilder(1, options, new AnalysisCacheImpl(), classHierarchy, scope);
            CallGraph callGraph = generateCallGraph(options, builder);


            PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();

            SDG<InstanceKey> sdg = createSDG(callGraph, pa);
            writeSDGResults(sdg, i);

            ControlFlowGraph<SSAInstruction, ISSABasicBlock> cfg = createCFG(callGraph);
            writeCFGResults(cfg, i);

            ArrayList<PDG> pdgs = createPDG(sdg, callGraph);
            writePDGResults(pdgs, i);

            writeCallGraphResults(callGraph, i);
        }

    }

    private static URL getFilesForResource(int index) {
        return Representations.class.getResource("train/train_" + index + ".jar");
    }

    private static CallGraph generateCallGraph(AnalysisOptions options, SSAPropagationCallGraphBuilder builder) throws CancelException {
        CallGraph callGraph = builder.makeCallGraph(options);
        return callGraph;
    }

    private static SDG createSDG(CallGraph callGraph, PointerAnalysis<InstanceKey> pa) {
        SDG<InstanceKey> sdg = new SDG(callGraph, pa, Slicer.DataDependenceOptions.NO_BASE_NO_HEAP_NO_EXCEPTIONS,
                Slicer.ControlDependenceOptions.NO_EXCEPTIONAL_EDGES);

        return sdg;
    }

    private static ControlFlowGraph<SSAInstruction, ISSABasicBlock> createCFG(CallGraph callGraph) {
        CGNode mainNode = callGraph.getNode(0);
        IR ir = mainNode.getIR();
        return ir.getControlFlowGraph();
    }

    private static ArrayList<PDG> createPDG(SDG sdg, CallGraph callGraph) {

        ArrayList<PDG> pdgs = new ArrayList<>();
        for (int i = 0; i < callGraph.getNumberOfNodes() - 1; i++) {
            CGNode mainNode = callGraph.getNode(i);
            PDG pdg = sdg.getPDG(mainNode);
            pdgs.add(pdg);
        }

        return pdgs;
    }

    private static void writeSDGResults(SDG sdg, int index) throws IOException {
        FileWriter sdgEdgelist = new FileWriter("./sdgs/sdg_" + index+ ".edgelist");
        for (int i = 0; i <= sdg.getNumberOfNodes() -1; i++) {
            Object bb = sdg.getNode(i);
            String currentBB = format("%d", i);
            if (bb != null) {
                Iterator<Statement> succNodes = sdg.getSuccNodes(bb);
                while (succNodes.hasNext()) {
                    String nextBB = format("%d", succNodes.next().getNode().getGraphNodeId());
                    sdgEdgelist.write(format("%s %s \n", currentBB, nextBB));
                }
            }
        }


    }

    private static void writePDGResults(ArrayList<PDG> pdgs, int index) throws IOException {
        FileWriter pdgEdgelist = new FileWriter("./pdgs/pdg_" + index + ".edgelist");
        for(int k = 0; k <= pdgs.size() - 1; k++) {
            PDG pdg = pdgs.get(k);
            for (int i = 0; i <= pdg.getNumberOfNodes() -1; i++) {
                Statement bb = pdg.getNode(i);
                String currentBB = format("%d", i);
                Iterator<Statement> succNodes = pdg.getSuccNodes(bb);
                while (succNodes.hasNext()) {
                    String nextBB = format("%d", succNodes.next().getNode().getGraphNodeId());
                    pdgEdgelist.write(format("%s %s \n", currentBB, nextBB));
                }
            }
        }

    }

    private static void writeCFGResults(ControlFlowGraph<SSAInstruction, ISSABasicBlock> cfg, int index) throws IOException {
        FileWriter cfgEdgelist = new FileWriter("./cfgs/cfg_" + index + ".edgelist");
        for (int i = 0; i <= cfg.getNumber(cfg.exit()); i++) {
            ISSABasicBlock bb = cfg.getNode(i);
            String currentBB = format("%d", i);
            Iterator<ISSABasicBlock> succNodes = cfg.getSuccNodes(bb);
            while (succNodes.hasNext()) {
                String nextBB = format("%d", succNodes.next().getNumber());
                cfgEdgelist.write(format("%s %s \n", currentBB, nextBB));
            }
        }
        cfgEdgelist.close();
    }

    private static void writeCallGraphResults(CallGraph callGraph, int index) throws IOException {
        FileWriter cgEdgelist = new FileWriter("./cgs/cg_" + index + ".edgelist");
        for (int i = 0; i <= callGraph.getNumberOfNodes() -1; i++) {
            CGNode bb = callGraph.getNode(i);
            String currentBB = format("%d", i);
            Iterator<CGNode> succNodes = callGraph.getSuccNodes(bb);
            while (succNodes.hasNext()) {
                String nextBB = format("%d", succNodes.next().getGraphNodeId());
                cgEdgelist.write(format("%s %s \n", currentBB, nextBB));
            }
        }

    }

}