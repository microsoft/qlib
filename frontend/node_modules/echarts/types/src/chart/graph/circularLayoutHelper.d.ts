import GraphSeriesModel from './GraphSeries.js';
import { GraphNode } from '../../data/Graph.js';
/**
 * `basedOn` can be:
 * 'value':
 *     This layout is not accurate and have same bad case. For example,
 *     if the min value is very smaller than the max value, the nodes
 *     with the min value probably overlap even though there is enough
 *     space to layout them. So we only use this approach in the as the
 *     init layout of the force layout.
 *     FIXME
 *     Probably we do not need this method any more but use
 *     `basedOn: 'symbolSize'` in force layout if
 *     delay its init operations to GraphView.
 * 'symbolSize':
 *     This approach work only if all of the symbol size calculated.
 *     That is, the progressive rendering is not applied to graph.
 *     FIXME
 *     If progressive rendering is applied to graph some day,
 *     probably we have to use `basedOn: 'value'`.
 */
export declare function circularLayout(seriesModel: GraphSeriesModel, basedOn: 'value' | 'symbolSize', draggingNode?: GraphNode, pointer?: [number, number]): void;
export declare function rotateNodeLabel(node: GraphNode, circularRotateLabel: boolean, cx: number, cy: number): void;
