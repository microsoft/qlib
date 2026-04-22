import SymbolDraw from '../helper/SymbolDraw.js';
import * as graphic from '../../util/graphic.js';
import { ECPolyline, ECPolygon } from './poly.js';
import ChartView from '../../view/Chart.js';
import LineSeriesModel, { LineSeriesOption } from './LineSeries.js';
import type GlobalModel from '../../model/Global.js';
import type ExtensionAPI from '../../core/ExtensionAPI.js';
import Cartesian2D from '../../coord/cartesian/Cartesian2D.js';
import Polar from '../../coord/polar/Polar.js';
import type SeriesData from '../../data/SeriesData.js';
import type { Payload, DisplayState, LabelOption } from '../../util/types.js';
import { CoordinateSystemClipArea } from '../../coord/CoordinateSystem.js';
import Model from '../../model/Model.js';
declare type PolarArea = ReturnType<Polar['getArea']>;
declare type Cartesian2DArea = ReturnType<Cartesian2D['getArea']>;
interface EndLabelAnimationRecord {
    lastFrameIndex: number;
    originalX?: number;
    originalY?: number;
}
declare class LineView extends ChartView {
    static readonly type = "line";
    _symbolDraw: SymbolDraw;
    _lineGroup: graphic.Group;
    _coordSys: Cartesian2D | Polar;
    _endLabel: graphic.Text;
    _polyline: ECPolyline;
    _polygon: ECPolygon;
    _stackedOnPoints: ArrayLike<number>;
    _points: ArrayLike<number>;
    _step: LineSeriesOption['step'];
    _valueOrigin: LineSeriesOption['areaStyle']['origin'];
    _clipShapeForSymbol: CoordinateSystemClipArea;
    _data: SeriesData;
    init(): void;
    render(seriesModel: LineSeriesModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    private packEventData;
    highlight(seriesModel: LineSeriesModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void;
    downplay(seriesModel: LineSeriesModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void;
    _changePolyState(toState: DisplayState): void;
    _newPolyline(points: ArrayLike<number>): ECPolyline;
    _newPolygon(points: ArrayLike<number>, stackedOnPoints: ArrayLike<number>): ECPolygon;
    _initSymbolLabelAnimation(data: SeriesData, coordSys: Polar | Cartesian2D, clipShape: PolarArea | Cartesian2DArea): void;
    _initOrUpdateEndLabel(seriesModel: LineSeriesModel, coordSys: Cartesian2D, inheritColor: string): void;
    _endLabelOnDuring(percent: number, clipRect: graphic.Rect, data: SeriesData, animationRecord: EndLabelAnimationRecord, valueAnimation: boolean, endLabelModel: Model<LabelOption>, coordSys: Cartesian2D): void;
    /**
     * @private
     */
    _doUpdateAnimation(data: SeriesData, stackedOnPoints: ArrayLike<number>, coordSys: Cartesian2D | Polar, api: ExtensionAPI, step: LineSeriesOption['step'], valueOrigin: LineSeriesOption['areaStyle']['origin'], connectNulls: boolean): void;
    remove(ecModel: GlobalModel): void;
}
export default LineView;
