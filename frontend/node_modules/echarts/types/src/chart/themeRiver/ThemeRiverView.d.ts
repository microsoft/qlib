import ChartView from '../../view/Chart.js';
import ThemeRiverSeriesModel from './ThemeRiverSeries.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
declare class ThemeRiverView extends ChartView {
    static readonly type = "themeRiver";
    readonly type = "themeRiver";
    private _layersSeries;
    private _layers;
    render(seriesModel: ThemeRiverSeriesModel, ecModel: GlobalModel, api: ExtensionAPI): void;
}
export default ThemeRiverView;
