import type GlobalModel from '../model/Global.js';
import type ExtensionAPI from './ExtensionAPI.js';
import type { CoordinateSystemCreator, CoordinateSystemMaster } from '../coord/CoordinateSystem.js';
declare class CoordinateSystemManager {
    private _coordinateSystems;
    create(ecModel: GlobalModel, api: ExtensionAPI): void;
    update(ecModel: GlobalModel, api: ExtensionAPI): void;
    getCoordinateSystems(): CoordinateSystemMaster[];
    static register: (type: string, creator: CoordinateSystemCreator) => void;
    static get: (type: string) => CoordinateSystemCreator;
}
export default CoordinateSystemManager;
