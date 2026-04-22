import ComponentView from '../../view/Component.js';
import RadarModel from '../../coord/radar/RadarModel.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
declare class RadarView extends ComponentView {
    static type: string;
    type: string;
    render(radarModel: RadarModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    _buildAxes(radarModel: RadarModel): void;
    _buildSplitLineAndArea(radarModel: RadarModel): void;
}
export default RadarView;
