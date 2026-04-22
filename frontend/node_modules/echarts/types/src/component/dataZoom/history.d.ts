import GlobalModel from '../../model/Global.js';
import { Dictionary } from '../../util/types.js';
import { DataZoomPayloadBatchItem } from './helper.js';
export declare type DataZoomStoreSnapshot = Dictionary<DataZoomPayloadBatchItem>;
/**
 * @param ecModel
 * @param newSnapshot key is dataZoomId
 */
export declare function push(ecModel: GlobalModel, newSnapshot: DataZoomStoreSnapshot): void;
export declare function pop(ecModel: GlobalModel): DataZoomStoreSnapshot;
export declare function clear(ecModel: GlobalModel): void;
export declare function count(ecModel: GlobalModel): number;
