import ChartView from '../../view/Chart.js';
import GaugeSeriesModel from './GaugeSeries.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import { ColorString } from '../../util/types.js';
interface PosInfo {
    cx: number;
    cy: number;
    r: number;
}
declare class GaugeView extends ChartView {
    static type: "gauge";
    type: "gauge";
    private _data;
    private _progressEls;
    private _titleEls;
    private _detailEls;
    render(seriesModel: GaugeSeriesModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    dispose(): void;
    _renderMain(seriesModel: GaugeSeriesModel, ecModel: GlobalModel, api: ExtensionAPI, colorList: [number, ColorString][], posInfo: PosInfo): void;
    _renderTicks(seriesModel: GaugeSeriesModel, ecModel: GlobalModel, api: ExtensionAPI, getColor: (percent: number) => ColorString, posInfo: PosInfo, startAngle: number, endAngle: number, clockwise: boolean, axisLineWidth: number): void;
    _renderPointer(seriesModel: GaugeSeriesModel, ecModel: GlobalModel, api: ExtensionAPI, getColor: (percent: number) => ColorString, posInfo: PosInfo, startAngle: number, endAngle: number, clockwise: boolean, axisLineWidth: number): void;
    _renderAnchor(seriesModel: GaugeSeriesModel, posInfo: PosInfo): void;
    _renderTitleAndDetail(seriesModel: GaugeSeriesModel, ecModel: GlobalModel, api: ExtensionAPI, getColor: (percent: number) => ColorString, posInfo: PosInfo): void;
}
export default GaugeView;
