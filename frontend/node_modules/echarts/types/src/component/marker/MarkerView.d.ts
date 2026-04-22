import ComponentView from '../../view/Component.js';
import { HashMap } from 'zrender/lib/core/util.js';
import MarkerModel from './MarkerModel.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import SeriesModel from '../../model/Series.js';
import Group from 'zrender/lib/graphic/Group.js';
interface MarkerDraw {
    group: Group;
}
declare abstract class MarkerView extends ComponentView {
    static type: string;
    type: string;
    /**
     * Markline grouped by series
     */
    markerGroupMap: HashMap<MarkerDraw>;
    init(): void;
    render(markerModel: MarkerModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    markKeep(drawGroup: MarkerDraw): void;
    toggleBlurSeries(seriesModelList: SeriesModel[], isBlur: boolean): void;
    abstract renderSeries(seriesModel: SeriesModel, markerModel: MarkerModel, ecModel: GlobalModel, api: ExtensionAPI): void;
}
export default MarkerView;
