import SeriesModel from '../../model/Series.js';
import SeriesData from '../../data/SeriesData.js';
import { SeriesOption, SeriesOnCartesianOptionMixin, SeriesOnGeoOptionMixin, SeriesOnPolarOptionMixin, SeriesOnCalendarOptionMixin, SeriesLargeOptionMixin, LineStyleOption, OptionDataValue, StatesOptionMixin, SeriesLineLabelOption, DimensionDefinitionLoose, DefaultStatesMixinEmphasis, ZRColor, CallbackDataParams } from '../../util/types.js';
import GlobalModel from '../../model/Global.js';
import type { LineDrawModelOption } from '../helper/LineDraw.js';
declare type LinesCoords = number[][];
declare type LinesValue = OptionDataValue | OptionDataValue[];
interface LinesLineStyleOption<TClr> extends LineStyleOption<TClr> {
    curveness?: number;
}
interface LinesStatesMixin {
    emphasis?: DefaultStatesMixinEmphasis;
}
export interface LinesStateOption<TCbParams = never> {
    lineStyle?: LinesLineStyleOption<(TCbParams extends never ? never : (params: TCbParams) => ZRColor) | ZRColor>;
    label?: SeriesLineLabelOption;
}
export interface LinesDataItemOption extends LinesStateOption, StatesOptionMixin<LinesStateOption, LinesStatesMixin> {
    name?: string;
    fromName?: string;
    toName?: string;
    symbol?: string[] | string;
    symbolSize?: number[] | number;
    coords?: LinesCoords;
    value?: LinesValue;
    effect?: LineDrawModelOption['effect'];
}
export interface LinesSeriesOption extends SeriesOption<LinesStateOption, LinesStatesMixin>, LinesStateOption<CallbackDataParams>, SeriesOnCartesianOptionMixin, SeriesOnGeoOptionMixin, SeriesOnPolarOptionMixin, SeriesOnCalendarOptionMixin, SeriesLargeOptionMixin {
    type?: 'lines';
    coordinateSystem?: string;
    symbol?: string[] | string;
    symbolSize?: number[] | number;
    effect?: LineDrawModelOption['effect'];
    /**
     * If lines are polyline
     * polyline not support curveness, label, animation
     */
    polyline?: boolean;
    /**
     * If clip the overflow.
     * Available when coordinateSystem is cartesian or polar.
     */
    clip?: boolean;
    data?: LinesDataItemOption[] | ArrayLike<number>;
    dimensions?: DimensionDefinitionLoose | DimensionDefinitionLoose[];
}
declare class LinesSeriesModel extends SeriesModel<LinesSeriesOption> {
    static readonly type = "series.lines";
    readonly type = "series.lines";
    static readonly dependencies: string[];
    visualStyleAccessPath: string;
    visualDrawType: "stroke";
    private _flatCoords;
    private _flatCoordsOffset;
    init(option: LinesSeriesOption): void;
    mergeOption(option: LinesSeriesOption): void;
    appendData(params: Pick<LinesSeriesOption, 'data'>): void;
    _getCoordsFromItemModel(idx: number): LinesCoords | (LinesDataItemOption & any[]);
    getLineCoordsCount(idx: number): number;
    getLineCoords(idx: number, out: number[][]): number;
    _processFlatCoordsArray(data: LinesSeriesOption['data']): {
        flatCoordsOffset: Uint32Array;
        flatCoords: Float64Array;
        count: number;
    };
    getInitialData(option: LinesSeriesOption, ecModel: GlobalModel): SeriesData<this, import("../../data/SeriesData").DefaultDataVisual>;
    formatTooltip(dataIndex: number, multipleSeries: boolean, dataType: string): string | import("../../component/tooltip/tooltipMarkup").TooltipMarkupNameValueBlock;
    preventIncremental(): boolean;
    getProgressive(): number | false;
    getProgressiveThreshold(): number;
    getZLevelKey(): string;
    static defaultOption: LinesSeriesOption;
}
export default LinesSeriesModel;
