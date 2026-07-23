import SeriesModel from '../../model/Series.js';
import Model from '../../model/Model.js';
import { SeriesOption, BoxLayoutOptionMixin, OptionDataValue, SeriesLabelOption, ItemStyleOption, LineStyleOption, LayoutOrient, ColorString, StatesOptionMixin, OptionDataItemObject, GraphEdgeItemObject, OptionDataValueNumeric, DefaultEmphasisFocus, CallbackDataParams } from '../../util/types.js';
import GlobalModel from '../../model/Global.js';
import SeriesData from '../../data/SeriesData.js';
import { LayoutRect } from '../../util/layout.js';
declare type FocusNodeAdjacency = boolean | 'inEdges' | 'outEdges' | 'allEdges';
export interface SankeyNodeStateOption<TCbParams = never> {
    label?: SeriesLabelOption;
    itemStyle?: ItemStyleOption<TCbParams>;
}
export interface SankeyEdgeStateOption {
    lineStyle?: SankeyEdgeStyleOption;
}
interface SankeyBothStateOption<TCbParams> extends SankeyNodeStateOption<TCbParams>, SankeyEdgeStateOption {
}
interface SankeyEdgeStyleOption extends LineStyleOption {
    curveness?: number;
}
interface ExtraStateOption {
    emphasis?: {
        focus?: DefaultEmphasisFocus | 'adjacency' | 'trajectory';
    };
}
export interface SankeyNodeItemOption extends SankeyNodeStateOption, StatesOptionMixin<SankeyNodeStateOption, ExtraStateOption>, OptionDataItemObject<OptionDataValue> {
    id?: string;
    localX?: number;
    localY?: number;
    depth?: number;
    draggable?: boolean;
    focusNodeAdjacency?: FocusNodeAdjacency;
}
export interface SankeyEdgeItemOption extends SankeyEdgeStateOption, StatesOptionMixin<SankeyEdgeStateOption, ExtraStateOption>, GraphEdgeItemObject<OptionDataValueNumeric> {
    focusNodeAdjacency?: FocusNodeAdjacency;
    edgeLabel?: SeriesLabelOption;
}
export interface SankeyLevelOption extends SankeyNodeStateOption, SankeyEdgeStateOption {
    depth: number;
}
export interface SankeySeriesOption extends SeriesOption<SankeyBothStateOption<CallbackDataParams>, ExtraStateOption>, SankeyBothStateOption<CallbackDataParams>, BoxLayoutOptionMixin {
    type?: 'sankey';
    /**
     * color will be linear mapped.
     */
    color?: ColorString[];
    coordinateSystem?: 'view';
    orient?: LayoutOrient;
    /**
     * The width of the node
     */
    nodeWidth?: number;
    /**
     * The vertical distance between two nodes
     */
    nodeGap?: number;
    /**
     * Control if the node can move or not
     */
    draggable?: boolean;
    /**
     * Will be allEdges if true.
     * @deprecated
     */
    focusNodeAdjacency?: FocusNodeAdjacency;
    /**
     * The number of iterations to change the position of the node
     */
    layoutIterations?: number;
    nodeAlign?: 'justify' | 'left' | 'right';
    data?: SankeyNodeItemOption[];
    nodes?: SankeyNodeItemOption[];
    edges?: SankeyEdgeItemOption[];
    links?: SankeyEdgeItemOption[];
    levels?: SankeyLevelOption[];
    edgeLabel?: SeriesLabelOption & {
        position?: 'inside';
    };
}
declare class SankeySeriesModel extends SeriesModel<SankeySeriesOption> {
    static readonly type = "series.sankey";
    readonly type = "series.sankey";
    levelModels: Model<SankeyLevelOption>[];
    layoutInfo: LayoutRect;
    /**
     * Init a graph data structure from data in option series
     */
    getInitialData(option: SankeySeriesOption, ecModel: GlobalModel): SeriesData<Model<any>, import("../../data/SeriesData").DefaultDataVisual>;
    setNodePosition(dataIndex: number, localPosition: number[]): void;
    /**
     * Return the graphic data structure
     *
     * @return graphic data structure
     */
    getGraph(): import("../../data/Graph").default;
    /**
     * Get edge data of graphic data structure
     *
     * @return data structure of list
     */
    getEdgeData(): SeriesData<Model<any>, import("../../data/SeriesData").DefaultDataVisual>;
    formatTooltip(dataIndex: number, multipleSeries: boolean, dataType: 'node' | 'edge'): import("../../component/tooltip/tooltipMarkup").TooltipMarkupNameValueBlock;
    optionUpdated(): void;
    getDataParams(dataIndex: number, dataType: 'node' | 'edge'): CallbackDataParams;
    static defaultOption: SankeySeriesOption;
}
export default SankeySeriesModel;
