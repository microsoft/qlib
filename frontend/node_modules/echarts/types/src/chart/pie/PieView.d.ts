import ChartView from '../../view/Chart.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import { Payload } from '../../util/types.js';
import PieSeriesModel from './PieSeries.js';
declare class PieView extends ChartView {
    static type: string;
    ignoreLabelLineUpdate: boolean;
    private _data;
    private _emptyCircleSector;
    render(seriesModel: PieSeriesModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void;
    dispose(): void;
    containPoint(point: number[], seriesModel: PieSeriesModel): boolean;
}
export default PieView;
