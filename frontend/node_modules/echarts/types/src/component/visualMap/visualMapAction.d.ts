import { Payload } from '../../util/types.js';
import GlobalModel from '../../model/Global.js';
export declare const visualMapActionInfo: {
    type: string;
    event: string;
    update: string;
};
export declare const visualMapActionHander: (payload: Payload, ecModel: GlobalModel) => void;
