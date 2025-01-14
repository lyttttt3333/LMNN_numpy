import numpy as np


class LMNN:
    def __init__(
        self,
        mu=0.5,
        learning_rate=1e-6,
        k=6,
        epoch=2000,
    ):
        self.mu = mu
        self.learning_rate = learning_rate
        self.k=k
        self.epoch=epoch

    def fit(self, X, y):
        loss_seq=[0,1]
        self.batch_num=X.shape[0]
        self.input_dim=X.shape[-1]
        self.L=np.eye(self.input_dim)
        self.X, self.y = X, y
        running = True
        for i in range(self.epoch):
            # if the loss cannot be smaller or descent in a very low speed, then stop
            if running == False or (int(loss_seq[-1]*1000)==int(loss_seq[-2]*1000)): 
                break
            count=0
            # count the how many times the learning rate decrease in this epoch, if it larger than 100, then stop
            non_step=True
            loss_old,grad=self._train()
            self.L_old = self.L
            delta_L = self.learning_rate * grad
            self.L=self.L - delta_L 
            while non_step:
                delta_L = self.learning_rate * grad
                self.L=self.L - delta_L 
                loss, _=self._train()
                # compute grad and loss
                if loss_old > loss:
                    self.learning_rate=self.learning_rate*1.01
                    non_step=False
                else:
                    self.learning_rate=self.learning_rate*0.5
                    self.L=self.L_old
                    count+=1
                    non_step=True
                    if count==100:
                        running = False
                        break
            print(f"{i} epoch",loss_old)
            # document the loss after optimization
            loss_seq.append(loss_old)

    def _train(self,):

        X=self.X.reshape(-1,self.input_dim)
        y=self.y
        X_hat=self.transform(X)
        # get the transformed X through L
        self.dist_mat=self._pair_dist(X_hat,X_hat)
        # get the distance (after transform) between any two node i and j in form of a matrix
        # element (i,j) means the distance between x_i and x_j
        self.max_dist = np.max(self.dist_mat)
        # the max_distance between any two nodes
        self.mis_match=self._pair_label(y)
        # a adjacent matrix 
        # if j have the same label with i , (i,j) = 0 otherwise (i,j) = 1, i.e. different label is 1
        self.match = self.mis_match == 0
        # a adjacent matrix 
        # if j have the same label with i , (i,j) = 1 otherwise (i,j) = 0, i.e. same label is 1

        # to compute the nearest k with the same label, 
        # we just let the differen label nodes have distance higher than the max_distance,
        # so there is no possible to include the different label nodes 
        # then we can get the distance of k same-label-nodes efficiently
        weight_dist=self.dist_mat + self.mis_match*self.max_dist
        self.j_bound = np.sort(weight_dist,axis=-1)[:,1:self.k+1]

        grad=self._grad()
        loss=self._loss()
        return loss, grad

    def _pair_dist(self,X1,X2):
        ft = X1[:, None, :] - X2
        dist = np.linalg.norm(ft,axis=-1,ord=2) 
        return dist
    
    def _pair_label(self,y):
        label_match = y[:, None] - y
        label_match = np.abs(label_match)
        label_match[label_match == 0] = 0
        label_match[label_match != 0] = 1
        return label_match


    def _loss(self):
        loss_pull=np.sum(self.dist_mat*self.pull_indicator)
        loss_push0=np.sum((self.dist_mat+1)*self.push0_indicator)
        loss_push1=np.sum(self.dist_mat*self.push1_indicator)
        return ((1-self.mu)*loss_pull+self.mu*(loss_push0-loss_push1))
    
    def _grad(self):
        ''' compute grad of pull term'''

        dist_match_large=self.dist_mat+self.match*self.max_dist
        # a distance matrix where the same labels are large
        # so all the nodes we get in the following operations have different labels with x_i
        dist_mismatch_large=self.dist_mat+self.mis_match*self.max_dist
        # a distance matrix where different labels are large
        # so all the nodes we get in the following operations have the same labels with x_i

        self.pull_indicator = dist_mismatch_large <= (self.j_bound[:,-1][:,np.newaxis])
        # an adjacent matrix
        # (i,j) means we need to compute the grad matrix of term ||L(x_i-x_j)||**2 (i,j) times
        # here all (i,j) are 0 or 1, means whether it is the target (needs "pull") or not
        pull_grad=self._mini_batch_grad(self.pull_indicator)
        # get the term grad under the indication of the adjacent matrix

        ''' compute grad of push term'''

        self.push1_indicator=np.zeros([self.batch_num,self.batch_num])
        self.push0_indicator=np.zeros([self.batch_num,self.batch_num])

        # accumulate the grad of each target node in turn

        for k in range(self.k):

            push1_indicator = (dist_match_large <= (self.j_bound[:,k][:,np.newaxis]+1))
            # an adjacent matrix
            # (i,j) means we need to compute the grad matrix of term ||L(x_i-x_j)||**2 (i,j) times
            # here all (i,j) are 0 or 1, means whether it is the inposter (need "push") or not

            push0_indicator = np.sum(push1_indicator,axis=-1).reshape(-1,1)*self.pull_indicator
            # an adjacent matrix as decribed above
            # here all (i,j) are 0 or an int (can be larger than 1)
            # means how many inposter it have (so this term needs to be computed so many times)

            self.push1_indicator += push1_indicator
            self.push0_indicator += push0_indicator

        push1_grad=self._mini_batch_grad(self.push1_indicator)
        push0_grad=self._mini_batch_grad(self.push0_indicator)
        # get the grad of push term

        push_grad = push0_grad - push1_grad

        grad = (1-self.mu)*pull_grad + self.mu*push_grad

        return 2 * self.L @ grad
    
    def _mini_batch_grad(self,indicator):
        indices=np.argwhere(indicator!=0)
        # get the index from adjacent matrix where (i,j) is non-zero
        row_indices=indices[:,0]
        col_indices=indices[:,1]
        weight=indicator[row_indices,col_indices].reshape(-1,1)
        # for any (i,j), how many times the grad of term ||L(x_i-x_j)||**2 need to be computed
        row_ft=self.X[row_indices,:]
        col_ft=self.X[col_indices,:]
        ft=row_ft-col_ft
        grad=ft.transpose(1,0)@(weight*ft)
        # get the sum of grad of terms with weight
        return grad


    def predict(self,train_X, test_X, train_y):
        from collections import Counter
        train_X=self.transform(train_X)
        test_X=self.transform(test_X)
        dist_mat=self._pair_dist(test_X,train_X)
        j_bound = np.sort(dist_mat,axis=-1)[:,1:self.k+1]
        pull_indicator = dist_mat <= (j_bound[:,-1][:,np.newaxis])
        pred=[]
        for row in pull_indicator:  
            row = row.reshape(-1)
            vote = row * train_y
            row_counts = Counter(vote)  
            max_count = row_counts.most_common()[1][0]
            pred.append(max_count)  
        pred=np.array(pred)
        return pred
    
    def Euclidean_predict(self,train_X, test_X, train_y):
        from collections import Counter
        dist_mat=self._pair_dist(test_X,train_X)
        j_bound = np.sort(dist_mat,axis=-1)[:,1:self.k+1]
        pull_indicator = dist_mat <= (j_bound[:,-1][:,np.newaxis])
        pred=[]
        for row in pull_indicator:  
            row = row.reshape(-1)
            vote = row * train_y
            row_counts = Counter(vote)  
            max_count = row_counts.most_common()[1][0]
            pred.append(max_count)  
        pred=np.array(pred)
        return pred
    
    def energy_predict(self,train_X, test_X, train_y):

        # some preparations
        possible_label = np.unique(train_y)
        possible_label = sorted(possible_label)
        test_num=test_X.shape[0]
        train_num=train_X.shape[0]
        transform_train_X=self.transform(train_X)
        transform_test_X=self.transform(test_X)

        _, mis_match_label = self._energy_label(label_hy=possible_label,label_gt=train_y,
                           dim0_num=test_num,dim1_num=train_num)
        mis_match_label_ij = train_y[:,None] != train_y[None,:]


        tj_L_dist=self._pair_dist(transform_test_X,transform_train_X)
        # distance mat in learned L
        tj_E_dist=self._pair_dist(test_X,train_X)
        # distance mat in Euclidean
        tj_E_dist=tj_E_dist[:,:,None]
        transformed_E_dist = tj_E_dist + mis_match_label * 10000
        bound_tj_indice=np.argsort(transformed_E_dist,axis=1)[:,0:self.k,:]
        bound_tj=np.sort(transformed_E_dist,axis=1)[:,0:self.k,:]

        target_indicator = transformed_E_dist <= bound_tj[:,-1,:][:,None,:]
        L_bound_tj_list=[]
        for i in range(self.label_num):
            L_bound_tj=tj_L_dist[np.arange(test_num)[:,None],bound_tj_indice[:,:,i]][:,:,None]
            L_bound_tj_list.append(L_bound_tj)
        L_bound_tj=np.concatenate(L_bound_tj_list,axis=-1)
        # get the Dm(xj,xt) with k order
        

        term0_sum=target_indicator*tj_L_dist[:,:,None]
        term0_sum=np.sum(term0_sum,axis=1)

        ij_E_dist=self._pair_dist(train_X,train_X)
        ij_L_dist=self._pair_dist(transform_train_X,transform_train_X)
        transformed_E_dist = ij_E_dist + mis_match_label_ij * 10000
        bound_ij_indice=np.argsort(transformed_E_dist,axis=1)[:,1:self.k+1]
        L_bound_ij=ij_L_dist[np.arange(train_num)[:,None],bound_ij_indice]
        # get the Dm(xi,xj) with k order

        term1_sum=0
        term2_sum=0
        for i in range(self.k):
            term1_indicator=mis_match_label
            term1_mat=1+L_bound_tj[:,i,:][:,None,:]-tj_L_dist[:,:,None]
            term1_mat[term1_mat < 0] = 0
            term2_indicator=mis_match_label.transpose(1,0,2)
            term2_mat=(1+L_bound_ij[:,i][:,None] - tj_L_dist.transpose(1,0))[:,:,None]
            term2_mat[term2_mat < 0 ] = 0

            term1_sum+=term1_indicator*term1_mat
            term2_sum+=term2_indicator*term2_mat

        term1_sum=np.sum(term1_sum,axis=1)
        term2_sum=np.sum(term2_sum,axis=0)

        energy_sum=(1-self.mu)*term0_sum + self.mu*term1_sum + self.mu*term2_sum
        # get the energy value for every possible label 

        pred = np.argmin(energy_sum,axis=-1) + 1
        # get the smallest label by energy
        return pred
    
    def _energy_label(self,label_hy,label_gt,dim0_num,dim1_num):
        label_num=len(label_hy)
        self.label_num=label_num
        label_hy=np.array(label_hy)
        label_hy=label_hy[None,None,:]
        label_hy=np.tile(label_hy,(dim0_num,dim1_num,1))
        label_gt=label_gt[None,:,None]
        label_gt=np.tile(label_gt,(dim0_num,1,label_num))
        mis_match = label_gt != label_hy
        match = label_gt == label_hy
        return match, mis_match

    def transform(self, X):
        return X @ self.L