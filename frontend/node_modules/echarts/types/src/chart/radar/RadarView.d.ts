import ChartView from '../../view/Chart.js';
import RadarSeriesModel from './RadarSeries.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import GlobalModel from '../../model/Global.js';
declare class RadarView extends ChartView {
    static type: string;
    type: string;
    private _data;
    render(seriesModel: RadarSeriesModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    remove(): void;
}
export default RadarView;
