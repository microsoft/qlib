/**
 * Single coordinate system creator.
 */
import Single from './Single.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
/**
 * Create single coordinate system and inject it into seriesModel.
 */
declare function create(ecModel: GlobalModel, api: ExtensionAPI): Single[];
declare const singleCreator: {
    create: typeof create;
    dimensions: string[];
};
export default singleCreator;
