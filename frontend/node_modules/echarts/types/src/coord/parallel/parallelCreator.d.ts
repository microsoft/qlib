import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import { CoordinateSystemMaster } from '../CoordinateSystem.js';
declare function createParallelCoordSys(ecModel: GlobalModel, api: ExtensionAPI): CoordinateSystemMaster[];
declare const parallelCoordSysCreator: {
    create: typeof createParallelCoordSys;
};
export default parallelCoordSysCreator;
