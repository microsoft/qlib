import Polar from './Polar.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import GlobalModel from '../../model/Global.js';
declare const polarCreator: {
    dimensions: string[];
    create: (ecModel: GlobalModel, api: ExtensionAPI) => Polar[];
};
export default polarCreator;
