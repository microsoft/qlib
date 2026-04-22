import SeriesData from '../../data/SeriesData.js';
import { SeriesOption, SeriesOnCartesianOptionMixin, SeriesOnPolarOptionMixin, SeriesOnCalendarOptionMixin, SeriesOnGeoOptionMixin, SeriesOnSingleOptionMixin, OptionDataValue, RoamOptionMixin, SeriesLabelOption, ItemStyleOption, LineStyleOption, SymbolOptionMixin, BoxLayoutOptionMixin, Dictionary, SeriesLineLabelOption, StatesOptionMixin, GraphEdgeItemObject, OptionDataValueNumeric, CallbackDataParams, DefaultEmphasisFocus } from '../../util/types.js';
import SeriesModel from '../../model/Series.js';
import Graph from '../../data/Graph.js';
import GlobalModel from '../../model/Global.js';
import { VectorArray } from 'zrender/lib/core/vector.js';
import { ForceLayoutInstance } from './forceLayout.js';
import { LineDataVisual } from '../../visual/commonVisualTypes.js';
declare type GraphDataValue = OptionDataValue | OptionDataValue[];
interface GraphEdgeLineStyleOption extends LineStyleOption {
    curveness?: number;
}
export interface GraphNodeStateOption<TCbParams = never> {
    itemStyle?: ItemStyleOption<TCbParams>;
    label?: SeriesLabelOption;
}
interface ExtraEmphasisState {
    focus?: DefaultEmphasisFocus | 'adjacency';
}
interface GraphNodeStatesMixin {
    emphasis?: ExtraEmphasisState;
}
interface GraphEdgeStatesMixin {
    emphasis?: ExtraEmphasisState;
}
export interface GraphNodeItemOption extends SymbolOptionMixin, GraphNodeStateOption, StatesOptionMixin<GraphNodeStateOption, GraphNodeStatesMixin> {
    id?: string;
    name?: string;
    value?: GraphDataValue;
    /**
     * Fixed x position
     */
    x?: number;
    /**
     * Fixed y position
     */
    y?: number;
    /**
     * If this node is fixed during force layout.
     */
    fixed?: boolean;
    /**
     * Index or name of category
     */
    category?: number | string;
    draggable?: boolean;
    cursor?: string;
}
export interface GraphEdgeStateOption {
    lineStyle?: GraphEdgeLineStyleOption;
    label?: SeriesLineLabelOption;
}
export interface GraphEdgeItemOption extends GraphEdgeStateOption, StatesOptionMixin<GraphEdgeStateOption, GraphEdgeStatesMixin>, GraphEdgeItemObject<OptionDataValueNumeric> {
    value?: number;
    /**
     * Symbol of both line ends
     */
    symbol?: string | string[];
    symbolSize?: number | number[];
    ignoreForceLayout?: boolean;
}
export interface GraphCategoryItemOption extends SymbolOptionMixin, GraphNodeStateOption, StatesOptionMixin<GraphNodeStateOption, GraphNodeStatesMixin> {
    name?: string;
    value?: OptionDataValue;
}
export interface GraphSeriesOption extends SeriesOption<GraphNodeStateOption<CallbackDataParams>, GraphNodeStatesMixin>, SeriesOnCartesianOptionMixin, SeriesOnPolarOptionMixin, SeriesOnCalendarOptionMixin, SeriesOnGeoOptionMixin, SeriesOnSingleOptionMixin, SymbolOptionMixin<CallbackDataParams>, RoamOptionMixin, BoxLayoutOptionMixin {
    type?: 'graph';
    coordinateSystem?: string;
    legendHoverLink?: boolean;
    layout?: 'none' | 'force' | 'circular';
    data?: (GraphNodeItemOption | GraphDataValue)[];
    nodes?: (GraphNodeItemOption | GraphDataValue)[];
    edges?: GraphEdgeItemOption[];
    links?: GraphEdgeItemOption[];
    categories?: GraphCategoryItemOption[];
    /**
     * @deprecated
     */
    focusNodeAdjacency?: boolean;
    /**
     * Symbol size scale ratio in roam
     */
    nodeScaleRatio?: 0.6;
    draggable?: boolean;
    edgeSymbol?: string | string[];
    edgeSymbolSize?: number | number[];
    edgeLabel?: SeriesLineLabelOption;
    label?: SeriesLabelOption;
    itemStyle?: ItemStyleOption<CallbackDataParams>;
    lineStyle?: GraphEdgeLineStyleOption;
    emphasis?: {
        focus?: Exclude<GraphNodeItemOption['emphasis'], undefined>['focus'];
        scale?: boolean | number;
        label?: SeriesLabelOption;
        edgeLabel?: SeriesLabelOption;
        itemStyle?: ItemStyleOption;
        lineStyle?: LineStyleOption;
    };
    blur?: {
        label?: SeriesLabelOption;
        edgeLabel?: SeriesLabelOption;
        itemStyle?: ItemStyleOption;
        lineStyle?: LineStyleOption;
    };
    select?: {
        label?: SeriesLabelOption;
        edgeLabel?: SeriesLabelOption;
        itemStyle?: ItemStyleOption;
        lineStyle?: LineStyleOption;
    };
    circular?: {
        rotateLabel?: boolean;
    };
    force?: {
        initLayout?: 'circular' | 'none';
        repulsion?: number | number[];
        gravity?: number;
        friction?: number;
        edgeLength?: number | number[];
        layoutAnimation?: boolean;
    };
    /**
     * auto curveness for multiple edge, invalid when `lineStyle.curveness` is set
     */
    autoCurveness?: boolean | number | number[];
}
declare class GraphSeriesModel extends SeriesModel<GraphSeriesOption> {
    static readonly type = "series.graph";
    readonly type = "series.graph";
    static readonly dependencies: string[];
    private _categoriesData;
    private _categoriesModels;
    /**
     * Preserved points during layouting
     */
    preservedPoints?: Dictionary<VectorArray>;
    forceLayout?: ForceLayoutInstance;
    hasSymbolVisual: boolean;
    init(option: GraphSeriesOption): void;
    mergeOption(option: GraphSeriesOption): void;
    mergeDefaultAndTheme(option: GraphSeriesOption): void;
    getInitialData(option: GraphSeriesOption, ecModel: GlobalModel): SeriesData;
    getGraph(): Graph;
    getEdgeData(): SeriesData<GraphSeriesModel, LineDataVisual>;
    getCategoriesData(): SeriesData;
    formatTooltip(dataIndex: number, multipleSeries: boolean, dataType: string): import("../../component/tooltip/tooltipMarkup").TooltipMarkupSection | import("../../component/tooltip/tooltipMarkup").TooltipMarkupNameValueBlock;
    _updateCategoriesData(): void;
    setZoom(zoom: number): void;
    setCenter(center: number[]): void;
    isAnimationEnabled(): boolean;
    static defaultOption: GraphSeriesOption;
}
export default GraphSeriesModel;
