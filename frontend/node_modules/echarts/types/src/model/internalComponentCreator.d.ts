import GlobalModel from './Global.js';
import { ComponentOption, ComponentMainType } from '../util/types.js';
interface InternalOptionCreator {
    (ecModel: GlobalModel): ComponentOption[];
}
export declare function registerInternalOptionCreator(mainType: ComponentMainType, creator: InternalOptionCreator): void;
export declare function concatInternalOptions(ecModel: GlobalModel, mainType: ComponentMainType, newCmptOptionList: ComponentOption[]): ComponentOption[];
export {};
