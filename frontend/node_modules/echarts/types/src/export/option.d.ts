import type { GridOption as GridComponentOption } from '../coord/cartesian/GridModel.js';
import type { PolarOption as PolarComponentOption } from '../coord/polar/PolarModel.js';
import type { RadarOption as RadarComponentOption } from '../coord/radar/RadarModel.js';
import type { GeoOption as GeoComponentOption } from '../coord/geo/GeoModel.js';
import type { RadiusAxisOption as RadiusAxisComponentOption, AngleAxisOption as AngleAxisComponentOption } from '../coord/polar/AxisModel.js';
import type { XAXisOption as XAXisComponentOption, YAXisOption as YAXisComponentOption } from '../coord/cartesian/AxisModel.js';
import type { SingleAxisOption as SingleAxisComponentOption } from '../coord/single/AxisModel.js';
import type { ParallelAxisOption as ParallelAxisComponentOption } from '../coord/parallel/AxisModel.js';
import type { ParallelCoordinateSystemOption as ParallelComponentOption } from '../coord/parallel/ParallelModel.js';
import type { CalendarOption as CalendarComponentOption } from '../coord/calendar/CalendarModel.js';
import type { ToolboxOption } from '../component/toolbox/ToolboxModel.js';
import type { TooltipOption as TooltipComponentOption, TopLevelFormatterParams } from '../component/tooltip/TooltipModel.js';
import type { AxisPointerOption as AxisPointerComponentOption } from '../component/axisPointer/AxisPointerModel.js';
import type { BrushOption as BrushComponentOption } from '../component/brush/BrushModel.js';
import type { TitleOption as TitleComponentOption } from '../component/title/install.js';
import type { TimelineOption as TimelineComponentOption } from '../component/timeline/TimelineModel.js';
import type { SliderTimelineOption as TimelineSliderComponentOption } from '../component/timeline/SliderTimelineModel.js';
import type { LegendOption as PlainLegendComponentOption } from '../component/legend/LegendModel.js';
import type { ScrollableLegendOption as ScrollableLegendComponentOption } from '../component/legend/ScrollableLegendModel.js';
import type { SliderDataZoomOption as SliderDataZoomComponentOption } from '../component/dataZoom/SliderZoomModel.js';
import type { InsideDataZoomOption as InsideDataZoomComponentOption } from '../component/dataZoom/InsideZoomModel.js';
import type { ContinousVisualMapOption as ContinousVisualMapComponentOption } from '../component/visualMap/ContinuousModel.js';
import type { PiecewiseVisualMapOption as PiecewiseVisualMapComponentOption } from '../component/visualMap/PiecewiseModel.js';
import type { MarkLineOption as MarkLineComponentOption } from '../component/marker/MarkLineModel.js';
import type { MarkPointOption as MarkPointComponentOption } from '../component/marker/MarkPointModel.js';
import type { MarkAreaOption as MarkAreaComponentOption } from '../component/marker/MarkAreaModel.js';
import type { LineSeriesOption as LineSeriesOptionInner } from '../chart/line/LineSeries.js';
import type { BarSeriesOption as BarSeriesOptionInner } from '../chart/bar/BarSeries.js';
import type { ScatterSeriesOption as ScatterSeriesOptionInner } from '../chart/scatter/ScatterSeries.js';
import type { PieSeriesOption as PieSeriesOptionInner } from '../chart/pie/PieSeries.js';
import type { RadarSeriesOption as RadarSeriesOptionInner } from '../chart/radar/RadarSeries.js';
import type { MapSeriesOption as MapSeriesOptionInner } from '../chart/map/MapSeries.js';
import type { TreeSeriesOption as TreeSeriesOptionInner } from '../chart/tree/TreeSeries.js';
import type { TreemapSeriesOption as TreemapSeriesOptionInner } from '../chart/treemap/TreemapSeries.js';
import type { GraphSeriesOption as GraphSeriesOptionInner } from '../chart/graph/GraphSeries.js';
import type { GaugeSeriesOption as GaugeSeriesOptionInner } from '../chart/gauge/GaugeSeries.js';
import type { FunnelSeriesOption as FunnelSeriesOptionInner } from '../chart/funnel/FunnelSeries.js';
import type { ParallelSeriesOption as ParallelSeriesOptionInner } from '../chart/parallel/ParallelSeries.js';
import type { SankeySeriesOption as SankeySeriesOptionInner } from '../chart/sankey/SankeySeries.js';
import type { BoxplotSeriesOption as BoxplotSeriesOptionInner } from '../chart/boxplot/BoxplotSeries.js';
import type { CandlestickSeriesOption as CandlestickSeriesOptionInner } from '../chart/candlestick/CandlestickSeries.js';
import type { EffectScatterSeriesOption as EffectScatterSeriesOptionInner } from '../chart/effectScatter/EffectScatterSeries.js';
import type { LinesSeriesOption as LinesSeriesOptionInner } from '../chart/lines/LinesSeries.js';
import type { HeatmapSeriesOption as HeatmapSeriesOptionInner } from '../chart/heatmap/HeatmapSeries.js';
import type { PictorialBarSeriesOption as PictorialBarSeriesOptionInner } from '../chart/bar/PictorialBarSeries.js';
import type { ThemeRiverSeriesOption as ThemeRiverSeriesOptionInner } from '../chart/themeRiver/ThemeRiverSeries.js';
import type { SunburstSeriesOption as SunburstSeriesOptionInner } from '../chart/sunburst/SunburstSeries.js';
import type { CustomSeriesOption as CustomSeriesOptionInner, CustomSeriesRenderItemAPI, CustomSeriesRenderItemParams, CustomSeriesRenderItemReturn, CustomSeriesRenderItem } from '../chart/custom/CustomSeries.js';
import { GraphicComponentLooseOption as GraphicComponentOption } from '../component/graphic/GraphicModel.js';
import type { DatasetOption as DatasetComponentOption } from '../component/dataset/install.js';
import type { ToolboxBrushFeatureOption } from '../component/toolbox/feature/Brush.js';
import type { ToolboxDataViewFeatureOption } from '../component/toolbox/feature/DataView.js';
import type { ToolboxDataZoomFeatureOption } from '../component/toolbox/feature/DataZoom.js';
import type { ToolboxMagicTypeFeatureOption } from '../component/toolbox/feature/MagicType.js';
import type { ToolboxRestoreFeatureOption } from '../component/toolbox/feature/Restore.js';
import type { ToolboxSaveAsImageFeatureOption } from '../component/toolbox/feature/SaveAsImage.js';
import type { ToolboxFeatureOption } from '../component/toolbox/featureManager.js';
import type { ECBasicOption, SeriesTooltipOption, AriaOption as AriaComponentOption, TooltipFormatterCallback, LabelFormatterCallback, CallbackDataParams, AnimationDurationCallback, AnimationDelayCallback, AnimationDelayCallbackParam, LabelLayoutOptionCallbackParams, LabelLayoutOptionCallback, TooltipPositionCallback, TooltipPositionCallbackParams } from '../util/types.js';
interface ToolboxComponentOption extends ToolboxOption {
    feature?: {
        brush?: ToolboxBrushFeatureOption;
        dataView?: ToolboxDataViewFeatureOption;
        dataZoom?: ToolboxDataZoomFeatureOption;
        magicType?: ToolboxMagicTypeFeatureOption;
        restore?: ToolboxRestoreFeatureOption;
        saveAsImage?: ToolboxSaveAsImageFeatureOption;
        [key: string]: ToolboxFeatureOption | {
            [key: string]: any;
        } | undefined;
    };
}
export { SliderDataZoomComponentOption, InsideDataZoomComponentOption };
export declare type DataZoomComponentOption = SliderDataZoomComponentOption | InsideDataZoomComponentOption;
export { ContinousVisualMapComponentOption, PiecewiseVisualMapComponentOption };
export declare type VisualMapComponentOption = ContinousVisualMapComponentOption | PiecewiseVisualMapComponentOption;
export { PlainLegendComponentOption, ScrollableLegendComponentOption };
export declare type LegendComponentOption = PlainLegendComponentOption | ScrollableLegendComponentOption;
export { GridComponentOption, PolarComponentOption, RadarComponentOption, GeoComponentOption, XAXisComponentOption, YAXisComponentOption, SingleAxisComponentOption, RadiusAxisComponentOption, AngleAxisComponentOption, ParallelComponentOption, CalendarComponentOption, TooltipComponentOption, AxisPointerComponentOption, BrushComponentOption, TitleComponentOption, TimelineComponentOption, MarkLineComponentOption, MarkPointComponentOption, MarkAreaComponentOption, ToolboxComponentOption, GraphicComponentOption, AriaComponentOption, DatasetComponentOption };
declare type SeriesInjectedOption = {
    markArea?: MarkAreaComponentOption;
    markLine?: MarkLineComponentOption;
    markPoint?: MarkPointComponentOption;
    tooltip?: SeriesTooltipOption;
};
export declare type LineSeriesOption = LineSeriesOptionInner & SeriesInjectedOption;
export declare type BarSeriesOption = BarSeriesOptionInner & SeriesInjectedOption;
export declare type ScatterSeriesOption = ScatterSeriesOptionInner & SeriesInjectedOption;
export declare type PieSeriesOption = PieSeriesOptionInner & SeriesInjectedOption;
export declare type RadarSeriesOption = RadarSeriesOptionInner & SeriesInjectedOption;
export declare type MapSeriesOption = MapSeriesOptionInner & SeriesInjectedOption;
export declare type TreeSeriesOption = TreeSeriesOptionInner & SeriesInjectedOption;
export declare type TreemapSeriesOption = TreemapSeriesOptionInner & SeriesInjectedOption;
export declare type GraphSeriesOption = GraphSeriesOptionInner & SeriesInjectedOption;
export declare type GaugeSeriesOption = GaugeSeriesOptionInner & SeriesInjectedOption;
export declare type FunnelSeriesOption = FunnelSeriesOptionInner & SeriesInjectedOption;
export declare type ParallelSeriesOption = ParallelSeriesOptionInner & SeriesInjectedOption;
export declare type SankeySeriesOption = SankeySeriesOptionInner & SeriesInjectedOption;
export declare type BoxplotSeriesOption = BoxplotSeriesOptionInner & SeriesInjectedOption;
export declare type CandlestickSeriesOption = CandlestickSeriesOptionInner & SeriesInjectedOption;
export declare type EffectScatterSeriesOption = EffectScatterSeriesOptionInner & SeriesInjectedOption;
export declare type LinesSeriesOption = LinesSeriesOptionInner & SeriesInjectedOption;
export declare type HeatmapSeriesOption = HeatmapSeriesOptionInner & SeriesInjectedOption;
export declare type PictorialBarSeriesOption = PictorialBarSeriesOptionInner & SeriesInjectedOption;
export declare type ThemeRiverSeriesOption = ThemeRiverSeriesOptionInner & SeriesInjectedOption;
export declare type SunburstSeriesOption = SunburstSeriesOptionInner & SeriesInjectedOption;
export declare type CustomSeriesOption = CustomSeriesOptionInner & SeriesInjectedOption;
/**
 * A map from series 'type' to series option
 * It's used for declaration merging in echarts extensions.
 * For example:
 * ```ts
 * import echarts from 'echarts';
 * declare module 'echarts/types/dist/echarts' {
 *   interface RegisteredSeriesOption {
 *     wordCloud: WordCloudSeriesOption
 *   }
 * }
 * ```
 */
export interface RegisteredSeriesOption {
    line: LineSeriesOption;
    bar: BarSeriesOption;
    scatter: ScatterSeriesOption;
    pie: PieSeriesOption;
    radar: RadarSeriesOption;
    map: MapSeriesOption;
    tree: TreeSeriesOption;
    treemap: TreemapSeriesOption;
    graph: GraphSeriesOption;
    gauge: GaugeSeriesOption;
    funnel: FunnelSeriesOption;
    parallel: ParallelSeriesOption;
    sankey: SankeySeriesOption;
    boxplot: BoxplotSeriesOption;
    candlestick: CandlestickSeriesOption;
    effectScatter: EffectScatterSeriesOption;
    lines: LinesSeriesOption;
    heatmap: HeatmapSeriesOption;
    pictorialBar: PictorialBarSeriesOption;
    themeRiver: ThemeRiverSeriesOption;
    sunburst: SunburstSeriesOption;
    custom: CustomSeriesOption;
}
declare type Values<T> = T[keyof T];
export declare type SeriesOption = Values<RegisteredSeriesOption>;
export interface EChartsOption extends ECBasicOption {
    dataset?: DatasetComponentOption | DatasetComponentOption[];
    aria?: AriaComponentOption;
    title?: TitleComponentOption | TitleComponentOption[];
    grid?: GridComponentOption | GridComponentOption[];
    radar?: RadarComponentOption | RadarComponentOption[];
    polar?: PolarComponentOption | PolarComponentOption[];
    geo?: GeoComponentOption | GeoComponentOption[];
    angleAxis?: AngleAxisComponentOption | AngleAxisComponentOption[];
    radiusAxis?: RadiusAxisComponentOption | RadiusAxisComponentOption[];
    xAxis?: XAXisComponentOption | XAXisComponentOption[];
    yAxis?: YAXisComponentOption | YAXisComponentOption[];
    singleAxis?: SingleAxisComponentOption | SingleAxisComponentOption[];
    parallel?: ParallelComponentOption | ParallelComponentOption[];
    parallelAxis?: ParallelAxisComponentOption | ParallelAxisComponentOption[];
    calendar?: CalendarComponentOption | CalendarComponentOption[];
    toolbox?: ToolboxComponentOption | ToolboxComponentOption[];
    tooltip?: TooltipComponentOption | TooltipComponentOption[];
    axisPointer?: AxisPointerComponentOption | AxisPointerComponentOption[];
    brush?: BrushComponentOption | BrushComponentOption[];
    timeline?: TimelineComponentOption | TimelineSliderComponentOption;
    legend?: LegendComponentOption | (LegendComponentOption)[];
    dataZoom?: DataZoomComponentOption | (DataZoomComponentOption)[];
    visualMap?: VisualMapComponentOption | (VisualMapComponentOption)[];
    graphic?: GraphicComponentOption | GraphicComponentOption[];
    series?: SeriesOption | SeriesOption[];
    options?: EChartsOption[];
    baseOption?: EChartsOption;
}
export { AnimationDurationCallback, AnimationDelayCallback, AnimationDelayCallbackParam as AnimationDelayCallbackParams, LabelFormatterCallback, CallbackDataParams as DefaultLabelFormatterCallbackParams, LabelLayoutOptionCallbackParams, LabelLayoutOptionCallback, TooltipFormatterCallback as TooltipComponentFormatterCallback, TopLevelFormatterParams as TooltipComponentFormatterCallbackParams, TooltipPositionCallback as TooltipComponentPositionCallback, TooltipPositionCallbackParams as TooltipComponentPositionCallbackParams, CustomSeriesRenderItemParams, CustomSeriesRenderItemAPI, CustomSeriesRenderItemReturn, CustomSeriesRenderItem };
