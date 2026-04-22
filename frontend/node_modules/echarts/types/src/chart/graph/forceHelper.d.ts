import * as vec2 from 'zrender/lib/core/vector.js';
import { RectLike } from 'zrender/lib/core/BoundingRect.js';
interface InputNode {
    p?: vec2.VectorArray;
    fixed?: boolean;
    /**
     * Weight
     */
    w: number;
    /**
     * Repulsion
     */
    rep: number;
}
interface InputEdge {
    ignoreForceLayout?: boolean;
    n1: InputNode;
    n2: InputNode;
    /**
     * Distance
     */
    d: number;
}
interface LayoutCfg {
    gravity?: number;
    friction?: number;
    rect?: RectLike;
}
export declare function forceLayout<N extends InputNode, E extends InputEdge>(inNodes: N[], inEdges: E[], opts: LayoutCfg): {
    warmUp: () => void;
    setFixed: (idx: number) => void;
    setUnfixed: (idx: number) => void;
    /**
     * Before step hook
     */
    beforeStep: (cb: (nodes: N[], edges: E[]) => void) => void;
    /**
     * After step hook
     */
    afterStep: (cb: (nodes: N[], edges: E[], finished: boolean) => void) => void;
    /**
     * Some formulas were originally copied from "d3.js"
     * https://github.com/d3/d3/blob/b516d77fb8566b576088e73410437494717ada26/src/layout/force.js
     * with some modifications made for this project.
     * See the license statement at the head of this file.
     */
    step: (cb?: (finished: boolean) => void) => void;
};
export {};
