import { Dictionary } from 'zrender/lib/core/types.js';
import SeriesData from './SeriesData.js';
import Model from '../model/Model.js';
import Element from 'zrender/lib/Element.js';
import { DimensionLoose, ParsedValue } from '../util/types.js';
declare class Graph {
    type: 'graph';
    readonly nodes: GraphNode[];
    readonly edges: GraphEdge[];
    data: SeriesData;
    edgeData: SeriesData;
    /**
     * Whether directed graph.
     */
    private _directed;
    private _nodesMap;
    /**
     * @type {Object.<string, module:echarts/data/Graph.Edge>}
     * @private
     */
    private _edgesMap;
    constructor(directed?: boolean);
    /**
     * If is directed graph
     */
    isDirected(): boolean;
    /**
     * Add a new node
     */
    addNode(id: string | number, dataIndex?: number): GraphNode;
    /**
     * Get node by data index
     */
    getNodeByIndex(dataIndex: number): GraphNode;
    /**
     * Get node by id
     */
    getNodeById(id: string): GraphNode;
    /**
     * Add a new edge
     */
    addEdge(n1: GraphNode | number | string, n2: GraphNode | number | string, dataIndex?: number): GraphEdge;
    /**
     * Get edge by data index
     */
    getEdgeByIndex(dataIndex: number): GraphEdge;
    /**
     * Get edge by two linked nodes
     */
    getEdge(n1: string | GraphNode, n2: string | GraphNode): GraphEdge;
    /**
     * Iterate all nodes
     */
    eachNode<Ctx>(cb: (this: Ctx, node: GraphNode, idx: number) => void, context?: Ctx): void;
    /**
     * Iterate all edges
     */
    eachEdge<Ctx>(cb: (this: Ctx, edge: GraphEdge, idx: number) => void, context?: Ctx): void;
    /**
     * Breadth first traverse
     * Return true to stop traversing
     */
    breadthFirstTraverse<Ctx>(cb: (this: Ctx, node: GraphNode, fromNode: GraphNode) => boolean | void, startNode: GraphNode | string, direction: 'none' | 'in' | 'out', context?: Ctx): void;
    update(): void;
    /**
     * @return {module:echarts/data/Graph}
     */
    clone(): Graph;
}
declare class GraphNode {
    id: string;
    inEdges: GraphEdge[];
    outEdges: GraphEdge[];
    edges: GraphEdge[];
    hostGraph: Graph;
    dataIndex: number;
    __visited: boolean;
    constructor(id?: string, dataIndex?: number);
    /**
     * @return {number}
     */
    degree(): number;
    /**
     * @return {number}
     */
    inDegree(): number;
    /**
    * @return {number}
    */
    outDegree(): number;
    getModel<T = unknown>(): Model<T>;
    getModel<T = unknown, S extends keyof T = keyof T>(path: S): Model<T[S]>;
    getAdjacentDataIndices(): {
        node: number[];
        edge: number[];
    };
    getTrajectoryDataIndices(): {
        node: number[];
        edge: number[];
    };
}
declare class GraphEdge {
    /**
     * The first node. If directed graph, it represents the source node.
     */
    node1: GraphNode;
    /**
     * The second node. If directed graph, it represents the target node.
     */
    node2: GraphNode;
    dataIndex: number;
    hostGraph: Graph;
    constructor(n1: GraphNode, n2: GraphNode, dataIndex?: number);
    getModel<T = unknown>(): Model<T>;
    getModel<T = unknown, S extends keyof T = keyof T>(path: S): Model<T[S]>;
    getAdjacentDataIndices(): {
        node: number[];
        edge: number[];
    };
    getTrajectoryDataIndices(): {
        node: number[];
        edge: number[];
    };
}
interface GraphDataProxyMixin {
    getValue(dimension?: DimensionLoose): ParsedValue;
    setVisual(key: string | Dictionary<any>, value?: any): void;
    getVisual(key: string): any;
    setLayout(layout: any, merge?: boolean): void;
    getLayout(): any;
    getGraphicEl(): Element;
    getRawIndex(): number;
}
interface GraphEdge extends GraphDataProxyMixin {
}
interface GraphNode extends GraphDataProxyMixin {
}
export default Graph;
export { GraphNode, GraphEdge };
