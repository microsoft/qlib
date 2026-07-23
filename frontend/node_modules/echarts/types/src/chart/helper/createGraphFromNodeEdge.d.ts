import SeriesData from '../../data/SeriesData.js';
import Graph from '../../data/Graph.js';
import { OptionSourceDataOriginal, GraphEdgeItemObject, OptionDataValue, OptionDataItemObject } from '../../util/types.js';
import SeriesModel from '../../model/Series.js';
export default function createGraphFromNodeEdge(nodes: OptionSourceDataOriginal<OptionDataValue, OptionDataItemObject<OptionDataValue>>, edges: OptionSourceDataOriginal<OptionDataValue, GraphEdgeItemObject<OptionDataValue>>, seriesModel: SeriesModel, directed: boolean, beforeLink: (nodeData: SeriesData, edgeData: SeriesData) => void): Graph;
