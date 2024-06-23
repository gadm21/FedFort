


from utils import * 
from tensorflow.keras.models import clone_model
# import exists and mkdir from os.path
from os.path import exists, join
from os import mkdir
import time 


def new_aggregate(weights) : 
    avg_weights = []
    for layer_id in range(len(weights[0]) ): 
        avg_layer = np.mean([weights[i][layer_id] for i in range(len(weights))], axis = 0)
        avg_weights.append(avg_layer)

    return avg_weights


def new_subtract(weights1, weights2) :
    delta_weights = []
    for layer_id in range(len(weights1)) : 
        delta_layer = weights1[layer_id] - weights2[layer_id]
        delta_weights.append(delta_layer)
    return delta_weights



def subtract_weights(weights1, weights2) :
    delta_weights = []
    for layer_id in range(len(weights1)) : 
        delta_layer = weights1[layer_id] - weights2[layer_id]
        delta_weights.append(delta_layer)
    return delta_weights

class FedSGD : 

    def __init__(self, exp_path, clients_data, test_data, initial_model, args) : 
        self.exp_path = exp_path
        self.args = args
        self.clients_data = clients_data
        self.test_data = test_data
    
    
        self.all_samples, self.all_labels = [], []
        for c in range(len(clients_data)) :
            self.all_samples.append(clients_data[c][0])
            self.all_labels.append(clients_data[c][1])
        self.all_samples = np.concatenate(self.all_samples, axis=0)
        self.all_labels = np.concatenate(self.all_labels, axis=0)

        self.server_model = initial_model
        # clone initial model to all clients
        self.clients_models = []
        for c in range(len(clients_data) ) : 
            model = clone_model(initial_model)
            model = compile_model(model, args) 
            self.clients_models.append(model)
        self.losses, self.accs = [], [] 
    
    def run(self, rounds, local_epochs = 1) : 
        for r in range(rounds) : 
            print("FedSGD round : ", r)
            deltas = []
            for c in range(len(self.clients_data)) : 
                self.download_server_model(c)
                
                self.test()
                deltas.append(self.local_train(c, local_epochs))
                delta_agg = new_aggregate(deltas)
                delta_agg = new_subtract(self.server_model.get_weights(), delta_agg)

                self.update_server_model(delta_agg)


            if ((r+1) % 30) == 0 :
                print("Saving the model and stats")
                res_path = join(self.exp_path, 'epoch_' + str(r+1))
                if not exists(res_path) : 
                    mkdir(res_path)

                self.server_model = compile_model(self.server_model, self.args)
                
                loss_fnn = tf.keras.losses.CategoricalCrossentropy(reduction = 'none')
                train_preds, train_losses = model_stats(self.server_model, self.all_samples, self.all_labels, loss_fnn)
                test_preds, test_losses = model_stats(self.server_model, self.test_data[0], self.test_data[1], loss_fnn)

                # save model 
                model_path = join(res_path, 'model.h5')
                # self.server_model.save(model_path)

                # save stats
                train_preds_path = join(res_path, 'train_preds.npy')
                train_losses_path = join(res_path, 'train_losses.npy')
                test_preds_path = join(res_path, 'test_preds.npy')
                test_losses_path = join(res_path, 'test_losses.npy')
                np.save(train_preds_path, train_preds)
                np.save(train_losses_path, train_losses)
                np.save(test_preds_path, test_preds)
                np.save(test_losses_path, test_losses)
                np.save(test_losses_path, test_losses)
            
    
    def download_server_model(self, client_id) :
        self.clients_models[client_id].set_weights(self.server_model.get_weights())

 
    def update_server_model(self, delta_agg) :
        new_weights = subtract_weights(self.server_model.get_weights(), delta_agg)
        self.server_model.set_weights(new_weights)

    def local_train(self, client_id, local_epochs) :
        weights0 = self.clients_models[client_id].get_weights()
        self.clients_models[client_id] = compile_model(self.clients_models[client_id], self.args)
        train_keras_model(self.clients_models[client_id], self.clients_data[client_id], self.test_data, epochs=local_epochs, batch_size = self.args.batch_size, verbose=0)
        # delta = np.array(weights0) - np.array(self.clients_models[client_id].get_weights())
        delta = subtract_weights(weights0, self.clients_models[client_id].get_weights())
        return delta

    def aggregate(self, deltas) :
        stacked_deltas = np.stack(deltas, axis=0)
        avg_delta = np.mean(stacked_deltas, axis=0)
        return avg_delta

    def test(self) :
        self.server_model = compile_model(self.server_model, self.args)
        score = test_keras_model(self.server_model, self.test_data, verbose=0)
        self.losses.append(score[0])
        self.accs.append(score[1])
        return score


    def save_scores(self) : 
        acc_path = join(self.exp_path, 'accuracy' + '.npy')
        loss_path = join(self.exp_path, 'loss' + '.npy')
        # perm_path = join(self.exp_path, 'perm' + '.npy')
        np.save(acc_path, self.accs)
        np.save(loss_path, self.losses)
        # np.save(perm_path, self.args.perm)

    def load_scores(self) :
        accuracy_path = join(self.exp_path, 'accuracy' + '.npy')
        loss_path = join(self.exp_path, 'loss' + '.npy')
        self.accs = np.load(accuracy_path)
        self.losses = np.load(loss_path)

    def plot_accuracy(self) :
        plt.plot(self.accs)
        plt.show()
    
    def plot_loss(self) :
        plt.plot(self.loss)
        plt.show()




class FedAvg :

    def __init__(self, exp_path, clients_data, test_data, initial_model, args) : 
        self.exp_path = exp_path
        self.args = args
        self.clients_data = clients_data
        self.test_data = test_data
        self.server_model = initial_model

        self.all_samples, self.all_labels = [], []
        for c in range(len(clients_data)) :
            self.all_samples.append(clients_data[c][0])
            self.all_labels.append(clients_data[c][1])
        self.all_samples = np.concatenate(self.all_samples, axis=0)
        self.all_labels = np.concatenate(self.all_labels, axis=0)

        self.clients_models = []
        for c in range(len(clients_data) ) : 
            model = clone_model(initial_model)
            new_args = update_args_with_dict(args, {'use_dp': False})
            model = compile_model(model, new_args) 
            print("initial training done for client ", id)
            train_keras_model(model = model, train_data = clients_data[c], test_data = self.test_data, epochs=20, batch_size = self.args.batch_size, verbose=1)
            self.clients_models.append(model)
        self.losses, self.accs = [], []

    def run(self, rounds, local_epochs = 1) :

    
        for r in range(rounds) :
            t0 = time.time() 
            weights = [] 
            for c in range(len(self.clients_data)) : 
                self.download_server_model(c)
                new_weights = self.local_train(c, local_epochs)
                weights.append(new_weights) 

            weights_agg = new_aggregate(weights)
            self.update_server_model(weights_agg)
            loss, acc = self.test()


            # if ((r+1) % 30) == 0 :
            #     print("Saving the model and stats")
            #     res_path = join(self.exp_path, 'epoch_' + str(r+1))
            #     if not exists(res_path) : 
            #         mkdir(res_path)

            #     self.server_model = compile_model(self.server_model, self.args)
                
            #     loss_fnn = tf.keras.losses.CategoricalCrossentropy(reduction = 'none')
            #     train_preds, train_losses = model_stats(self.server_model, self.all_samples, self.all_labels, loss_fnn)
            #     test_preds, test_losses = model_stats(self.server_model, self.test_data[0], self.test_data[1], loss_fnn)

            #     # save model 
            #     model_path = join(res_path, 'model.h5')
            #     # self.server_model.save(model_path)

            #     # save stats
            #     train_preds_path = join(res_path, 'train_preds.npy')
            #     train_losses_path = join(res_path, 'train_losses.npy')
            #     test_preds_path = join(res_path, 'test_preds.npy')
            #     test_losses_path = join(res_path, 'test_losses.npy')
            #     np.save(train_preds_path, train_preds)
            #     np.save(train_losses_path, train_losses)
            #     np.save(test_preds_path, test_preds)
            #     np.save(test_losses_path, test_losses)
            #     np.save(test_losses_path, test_losses)
                

            t1 = time.time()
            print("FedAvg round {}, time:{}, accuracy:{} ".format(r, t1-t0, acc))
            # if len(self.accs ) > 11: 
            #     # check if accuracy is not improving
            #     if np.mean(np.subtract(self.accs[-10:], self.accs[-11:-1])) < 0.01:
            #         print("Breaking the training loop as I am not improving anymore :(")
            #         break
            return acc 
            

    def download_server_model(self, client_id) :
        self.clients_models[client_id].set_weights(self.server_model.get_weights())

    def local_train(self, client_id, local_epochs) :
        train_keras_model(self.clients_models[client_id], self.clients_data[client_id], self.test_data, epochs=local_epochs, batch_size = self.args.batch_size, verbose=0)
        return self.clients_models[client_id].get_weights()
    
    def aggregate(self, weights):
        stacked_weights = np.stack(weights, axis=0)
        weights_agg = np.mean(stacked_weights, axis=0)
        return weights_agg
    

    def update_server_model(self, weights_agg) :
        self.server_model.set_weights(weights_agg)
   
    def test(self) :
        self.server_model = compile_model(self.server_model, self.args)
        score = test_keras_model(self.server_model, self.test_data, verbose=0)
        self.losses.append(score[0])
        self.accs.append(score[1])
        return score[0], score[1]



    def save_scores(self) : 
        acc_path = join(self.exp_path, 'accuracy' + '.npy')
        loss_path = join(self.exp_path, 'loss' + '.npy')
        # perm_path = join(self.exp_path, 'perm' + '.npy')
        np.save(acc_path, self.accs)
        np.save(loss_path, self.losses)
        # np.save(perm_path, self.args.perm)
    
    def load_scores(self) :
        accuracy_path = join(self.exp_path, 'accuracy' + '.npy')
        loss_path = join(self.exp_path, 'loss' + '.npy')
        self.accs = np.load(accuracy_path)
        self.losses = np.load(loss_path)

    def plot_accuracy(self) :
        plt.plot(self.accs)
        plt.show()
    
    def plot_loss(self) :
        plt.plot(self.losses)
        plt.show()



class FedProx :

    def __init__(self, exp_path, clients_data, test_data, initial_model, args) : 
        self.exp_path = exp_path
        self.args = args
        self.clients_data = clients_data
        self.test_data = test_data


        self.all_samples, self.all_labels = [], []
        for c in range(len(clients_data)) :
            self.all_samples.append(clients_data[c][0])
            self.all_labels.append(clients_data[c][1])
        self.all_samples = np.concatenate(self.all_samples, axis=0)
        self.all_labels = np.concatenate(self.all_labels, axis=0)


        self.server_model = initial_model
        self.server_model = compile_model(self.server_model, args)

        self.core_loss_fn = tf.keras.losses.categorical_crossentropy
        
        self.clients_models = [clone_model(initial_model) for _ in range(len(clients_data))]
        # for model in self.clients_models :
        #     model = compile_model(model, args)
        self.losses, self.accs = [], []
        self.mu = args.mu


    def run(self, rounds, local_epochs = 1) :
        for r in range(rounds) : 
            print("FedProx round : ", r)
            weights = []
            for c in range(len(self.clients_data)) : 
                self.download_server_model(c)
                round_initial_weights = self.server_model.get_weights()
                reduce_mean = not self.args.use_dp 
                loss_fn = self.create_fedprox_loss(c, round_initial_weights, self.mu, reduce_mean = reduce_mean)
                self.clients_models[c] = compile_model(self.clients_models[c], self.args, loss_fn = loss_fn)
                new_weights = self.local_train(c, local_epochs)
                weights.append(new_weights)
            
            weights_agg = new_aggregate(weights)
            self.update_server_model(weights_agg)
            loss, acc = self.test()
            print("FedProx round {}, accuracy:{} ".format(r, acc))

            if ((r+1) % 30) == 0 :
                print("Saving the model and stats")
                res_path = join(self.exp_path, 'epoch_' + str(r+1))
                if not exists(res_path) : 
                    mkdir(res_path)

                self.server_model = compile_model(self.server_model, self.args)
                
                loss_fnn = tf.keras.losses.CategoricalCrossentropy(reduction = 'none')
                train_preds, train_losses = model_stats(self.server_model, self.all_samples, self.all_labels, loss_fnn)
                test_preds, test_losses = model_stats(self.server_model, self.test_data[0], self.test_data[1], loss_fnn)

                # save model 
                model_path = join(res_path, 'model.h5')
                # self.server_model.save(model_path)

                # save stats
                train_preds_path = join(res_path, 'train_preds.npy')
                train_losses_path = join(res_path, 'train_losses.npy')
                test_preds_path = join(res_path, 'test_preds.npy')
                test_losses_path = join(res_path, 'test_losses.npy')
                np.save(train_preds_path, train_preds)
                np.save(train_losses_path, train_losses)
                np.save(test_preds_path, test_preds)
                np.save(test_losses_path, test_losses)
                np.save(test_losses_path, test_losses)
        return acc
            

    def create_fedprox_loss(self, c, round_initial_weights, mu, reduce_mean = True):
        def fedprox_loss_fn(output, target):
            # Compute the standard cross-entropy loss
            
            if reduce_mean : 
                ce_loss = tf.keras.losses.CategoricalCrossentropy(reduction = 'none') (target, output)
            else : 
                ce_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(target, output))

            # Compute the FedProx regularization term
            l2_norm = 0
            for param_init, param in zip(round_initial_weights, self.clients_models[c].trainable_variables):
                l2_norm += tf.norm(param - param_init)

            # Compute the total loss with FedProx regularization
            loss = ce_loss + (mu / 2) * l2_norm

            return loss

        return fedprox_loss_fn


    def download_server_model(self, client_id) :
        self.clients_models[client_id].set_weights(self.server_model.get_weights())

    def local_train(self, client_id, local_epochs) :
        train_keras_model(self.clients_models[client_id], self.clients_data[client_id], self.test_data, epochs=local_epochs, batch_size = self.args.batch_size, verbose=0)
        return self.clients_models[client_id].get_weights()
    
    def aggregate(self, weights) :
        stacked_weights = np.stack(weights, axis=0)
        weights_agg = np.mean(stacked_weights, axis=0)
        return weights_agg 
    
    def update_server_model(self, weights_agg) :
        self.server_model.set_weights(weights_agg)

    def test(self) :
        self.server_model = compile_model(self.server_model, self.args)
        score = test_keras_model(self.server_model, self.test_data, verbose=0)
        self.losses.append(score[0])
        self.accs.append(score[1])
        return score[0], score[1]


    def save_scores(self) : 
        acc_path = join(self.exp_path, 'accuracy' + '.npy')
        loss_path = join(self.exp_path, 'loss' + '.npy')
        # perm_path = join(self.exp_path, 'perm' + '.npy')
        np.save(acc_path, self.accs)
        np.save(loss_path, self.losses)
        # np.save(perm_path, self.args.perm)
    
    def load_scores(self) :
        accuracy_path = join(self.exp_path, 'accuracy' + '.npy')
        loss_path = join(self.exp_path, 'loss' + '.npy')
        self.accs = np.load(accuracy_path)
        self.losses = np.load(loss_path)

    def plot_accuracy(self) :
        plt.plot(self.accs)
        plt.show()
    
    def plot_loss(self) :
        plt.plot(self.losses)
        plt.show()



class FedAKD:

    def __init__(self, exp_path, clients_data, test_data, proxy_data, clients_model_fn, args):
        
        self.exp_path = exp_path
        self.args = args
        self.clients_data = clients_data
        self.test_data = test_data
        self.proxy_data = proxy_data[0]
        self.proxy_labels = proxy_data[1]


        self.all_samples, self.all_labels = [], []
        for c in range(len(clients_data)) :
            self.all_samples.append(clients_data[c][0])
            self.all_labels.append(clients_data[c][1])
        self.all_samples = np.concatenate(self.all_samples, axis=0)
        self.all_labels = np.concatenate(self.all_labels, axis=0)


        self.temperature = args.temperature
        self.aalpha = args.aalpha # [0, inf)
        self.bbeta = args.bbeta  # (-inf, inf) seed
        
        self.clients_models = []
        self.smoothed_clients_models = [] 

        self.losses, self.local_loss = [], []
        self.accs, self.local_accuracy = [], []
        
        for id in range(len(clients_data)):
            args.client_id = id
            model = clients_model_fn(args, compile_model = False)     
            
            # initial training
            model.compile(
            loss="mean_absolute_error",
            optimizer="adam",
            metrics=['accuracy'])

            print("initial alignment done for client ", id)
            train_keras_model(model = model, train_data = (self.proxy_data, self.proxy_labels), test_data = None, epochs=20, batch_size = self.args.batch_size, verbose=1)
            new_args = update_args_with_dict(args, {'use_dp': False})
            model = compile_model(model, new_args) 
            print("initial training done for client ", id)
            train_keras_model(model = model, train_data = clients_data[c], test_data = self.test_data, epochs=20, batch_size = self.args.batch_size, verbose=1)
            
            
            self.clients_models.append(model)


    def mixup(self, x1, x2, alpha=0.2):
        l = np.random.beta(alpha, alpha)
        x_l = l * x1 + (1 - l) * x2
        return x_l


    def run(self, rounds, local_epochs = 1):


        for r in range(rounds): 
            t0 = time.time()
            print("FedAKD round : ", r)

            # 0. Mixup proxy data
            p = np.random.permutation(len(self.proxy_data))
            mixup_proxy_data = self.mixup(self.proxy_data, self.proxy_data[p], self.aalpha)

            all_soft_labels = []
            for c in range(len(self.clients_data)): 
                # 1. Local training on private data
                self.local_train(c, local_epochs)

                # 2. Create temperature scaled model using the weights of the local model
                t_model = self.create_temperature_scaled_model(self.clients_models[c], self.temperature)
                self.smoothed_clients_models.append(t_model)

                # 3. Create temperature smoothed labels
                t_smoothed_labels = t_model.predict(mixup_proxy_data)
                all_soft_labels.append(t_smoothed_labels)


            # 4. Aggregate the temperature smoothed labels
            aggregated_soft_labels = np.mean(all_soft_labels, axis=0)

            # 5. KD training using the proxy data and the aggregated soft labels
            for c in range(len(self.clients_data)):
                self.kd_train(c, mixup_proxy_data, aggregated_soft_labels, local_epochs)

                # 6. Update original client models with the new weights
                self.clients_models[c].set_weights(self.smoothed_clients_models[c].get_weights())
            
            kd_loss, kd_acc = self.test()
            t1 = time.time()
            print("FedAKD round {}, accuracy:{} ".format(r, kd_acc))
        return kd_acc
            




    def create_temperature_scaled_model(self, initial_model, temperature):
        temperature_scaled_model = clone_model(initial_model)
        temperature_scaled_model.set_weights(initial_model.get_weights())
        # temperature_scaled_model.pop()  # Remove the last softmax layer
        temperature_scaled_model.add(tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x / temperature)))
        temperature_scaled_model.compile(
            loss="mean_absolute_error",
            optimizer="adam",
            metrics=['accuracy']
        )
        return temperature_scaled_model

    def download_server_model(self, client_id):
        self.clients_models[client_id].set_weights(self.server_model.get_weights())

    def local_train(self, client_id, local_epochs):
        train_keras_model(self.clients_models[client_id], self.clients_data[client_id], self.test_data, epochs=local_epochs, batch_size = self.args.batch_size, verbose=0)
        return self.clients_models[client_id].get_weights()

    def kd_train(self, client_id, proxy_data, soft_labels, epochs):
        train_keras_model(self.smoothed_clients_models[client_id], (proxy_data, soft_labels), epochs=epochs, verbose=0)
        return self.smoothed_clients_models[client_id].get_weights()

    def aggregate(self, weights):
        stacked_weights = np.stack(weights, axis=0)
        weights_agg = np.mean(stacked_weights, axis=0)
        return weights_agg

    def update_server_model(self, weights_agg):
        self.server_model.set_weights(weights_agg)
        self.temperature_scaled_model.set_weights(weights_agg)

    def test(self):
        losses, accs = [], []
        for c in range(len(self.clients_data)):
            loss, acc = test_keras_model(self.clients_models[c], self.test_data, verbose=0)
            losses.append(loss)
            accs.append(acc)

        self.local_accuracy.append(accs)
        self.local_loss.append(losses)
        self.accs.append(np.mean(accs))
        self.losses.append(np.mean(losses))

        return self.losses[-1], self.accs[-1]


    def save_scores(self) : 
        acc_path = join(self.exp_path, 'accuracy' + '.npy')
        loss_path = join(self.exp_path, 'loss' + '.npy')
        # perm_path = join(self.exp_path, 'perm' + '.npy')
        np.save(acc_path, self.accs)
        np.save(loss_path, self.losses)
        # np.save(perm_path, self.args.perm)



def run_local(train_data, test_data, args, proxy_data = None) : 
    pass 


def run_central(model, train_data, test_data, args, proxy_data = None) : 
    if isinstance(train_data, list) : 
        X, y = [], [] 
        for data in train_data :
            X.append(data[0])
            y.append(data[1])
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        train_data = (X, y)
    
    t0 = time.time() 
    history = train_keras_model(model, train_data, test_data, epochs = args.rounds, batch_size = args.batch_size) 
    t1 = time.time()
    train_time = t1 - t0

    return model, history, train_time





def run_fedavg(model, train_data, test_data, args, proxy_data = None) : 

    fedavg = FedAvg(clients_data = train_data, test_data = test_data, initial_model = model, args = args)
    
    t0 = time.time() 
    fedavg.run(args.rounds, args.local_epochs) 
    t1 = time.time()
    train_time = t1 - t0

    return model, train_time



def run_fedakd(model, train_data, test_data, args, proxy_data) : 

    fedakd = FedAKD(clients_data = train_data, test_data = test_data, proxy_data = proxy_data, clients_model_fn = create_model_based_on_data, args = args)
    
    t0 = time.time() 
    fedakd.run(args.rounds, args.local_epochs) 
    t1 = time.time()
    train_time = t1 - t0

    return model, train_time
