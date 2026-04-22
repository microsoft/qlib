import SeriesModel from '../../model/Series.js';
import { SeriesOption, BoxLayoutOptionMixin, SeriesEncodeOptionMixin, OptionDataItemObject, OptionDataValueNumeric, ParsedValue, SeriesOnGeoOptionMixin, StatesOptionMixin, SeriesLabelOption, StatesMixinBase, CallbackDataParams } from '../../util/types.js';
import { Dictionary } from 'zrender/lib/core/types.js';
import GeoModel, { GeoCommonOptionMixin, GeoItemStyleOption } from '../../coord/geo/GeoModel.js';
import SeriesData from '../../data/SeriesData.js';
import Model from '../../model/Model.js';
import Geo from '../../coord/geo/Geo.js';
import { ECSymbol } from '../../util/symbol.js';
import { LegendIconParams } from '../../component/legend/LegendModel.js';
import { Group } from '../../util/graphic.js';
export interface MapStateOption<TCbParams = never> {
    itemStyle?: GeoItemStyleOption<TCbParams>;
    label?: SeriesLabelOption;
}
export interface MapDataItemOption extends MapStateOption, StatesOptionMixin<MapStateOption, StatesMixinBase>, OptionDataItemObject<OptionDataValueNumeric> {
    cursor?: string;
    silent?: boolean;
}
export declare type MapValueCalculationType = 'sum' | 'average' | 'min' | 'max';
export interface MapSeriesOption extends SeriesOption<MapStateOption<CallbackDataParams>, StatesMixinBase>, MapStateOption<CallbackDataParams>, GeoCommonOptionMixin, SeriesOnGeoOptionMixin, BoxLayoutOptionMixin, SeriesEncodeOptionMixin {
    type?: 'map';
    coordinateSystem?: string;
    silent?: boolean;
    markLine?: any;
    markPoint?: any;
    markArea?: any;
    mapValueCalculation?: MapValueCalculationType;
    showLegendSymbol?: boolean;
    geoCoord?: Dictionary<number[]>;
    data?: (OptionDataValueNumeric | OptionDataValueNumeric[] | MapDataItemOption)[];
    nameProperty?: string;
}
declare class MapSeries extends SeriesModel<MapSeriesOption> {
    static type: "series.map";
    type: "series.map";
    static dependencies: string[];
    static layoutMode: "box";
    coordinateSystem: Geo;
    originalData: SeriesData;
    mainSeries: MapSeries;
    needsDrawMap: boolean;
    seriesGroup: MapSeries[];
    getInitialData(this: MapSeries, option: MapSeriesOption): SeriesData;
    /**
     * If no host geo model, return null, which means using a
     * inner exclusive geo model.
     */
    getHostGeoModel(): GeoModel;
    getMapType(): string;
    getRawValue(dataIndex: number): ParsedValue;
    /**
     * Get model of region
     */
    getRegionModel(regionName: string): Model<MapDataItemOption>;
    /**
     * Map tooltip formatter
     */
    formatTooltip(dataIndex: number, multipleSeries: boolean, dataType: string): import("../../component/tooltip/tooltipMarkup").TooltipMarkupSection;
    getTooltipPosition: (this: MapSeries, dataIndex: number) => number[];
    setZoom(zoom: number): void;
    setCenter(center: number[]): void;
    getLegendIcon(opt: LegendIconParams): ECSymbol | Group;
    static defaultOption: MapSeriesOption;
}
export default MapSeries;
