/**
 * Count the number of edges between the same two points, used to obtain the curvature table and the parity of the edge
 * @see /graph/GraphSeries.js@getInitialData
 * @param {module:echarts/model/SeriesModel} seriesModel
 */
export declare function initCurvenessList(seriesModel: any): void;
/**
 * set edgeMap with key
 * @param {number|string|module:echarts/data/Graph.Node} n1
 * @param {number|string|module:echarts/data/Graph.Node} n2
 * @param {module:echarts/model/SeriesModel} seriesModel
 * @param {number} index
 */
export declare function createEdgeMapForCurveness(n1: any, n2: any, seriesModel: any, index: any): void;
/**
 * get curvature for edge
 * @param edge
 * @param {module:echarts/model/SeriesModel} seriesModel
 * @param index
 */
export declare function getCurvenessForEdge(edge: any, seriesModel: any, index: any, needReverse?: boolean): any;
