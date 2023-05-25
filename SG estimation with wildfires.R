library(readxl)
library(stringr)
library(ggplot2)
library(viridis)
library("PerformanceAnalytics")
setwd("~/MS Stats/Research/Wildfire/data/processed_data")
data <- read_excel("fire_modeling_20230222.xlsx")
test_data <- read_excel("random_points_weather_20230227.xlsx")



# Subset data to years used for modeling
modeling_years <- list("2008", "2009", "2010", "2011", "2012")
data <- subset(data, DISCOVER_YEAR %in% modeling_years)
data$constant = 1
data = data[complete.cases(data), ]
test_data$constant = 1
test_data <- subset(test_data, region_id %in% unique(data$region_id))

all_vars = c('BKGD_RT', 'AWND', 'TMAX', 'TMIN', 
             'EVAP', 'PRCP_ROLLING')

# Splitting the fires into natural and unnatural fires
natural_df = data[data$LGHTNG == 1,]
unnatural_df = data[data$LGHTNG == 0,]


### DATA VISUALIZATION

chart.Correlation(data[,all_vars], histogram=TRUE, pch=19)

# Function to create a heatmap of fire variables
viz_df = expand.grid(letters[1:10],0:9)
colnames(viz_df) <- c("latitude_bin","longitude_bin")
viz_df$region_id <- paste(viz_df$latitude_bin, viz_df$longitude_bin, sep="")

create_heatmap = function(viz_df, values_df, plotname, clrs = c("thistle2", "darkblue")){
  viz_df = merge(viz_df, values_df, by = 'region_id', all.x = TRUE)
  viz_df$longitude_bin <- str_pad(viz_df$longitude_bin, 2, pad = "0")
  p <- ggplot(viz_df, aes(x = longitude_bin, y = latitude_bin, fill = fires)) +
    geom_tile() +
    scale_fill_gradient(low=clrs[1], high=clrs[2], na.value="white")+
    #scale_fill_viridis(discrete=FALSE) +
    labs(fill='value')
  
  ggsave(paste0(plotname, '.png'), plot = last_plot(), device = "png", 
         path = "~/MS Stats/Research/Wildfire/results")
  print(p)}

plot_samples = function(data, vars, model, n_sample = 100){
  data$lam_est = as.matrix(data[, vars]) %*% model$par
  ggplot() + 
    geom_point(data = data[sample(nrow(data), 100), ], 
               aes(x = POO_LONGITUDE, y = POO_LATITUDE, size = lam_est))
}

# Visualizing where the fires occur across California
values_df = as.data.frame(table(data$region_id, dnn = list("region_id")), responseName = "fires")
create_heatmap(viz_df, values_df, "Fire Counts")

values_df = as.data.frame(table(natural_df$region_id, dnn = list("region_id")), responseName = "fires")
create_heatmap(viz_df, values_df, "Natural Fires")



### MODELING

# Capture the size of the areas
t = length(unique(data$DISCOVER_YEAR))
s = 65*71 #squared miles
s = s/621.371^2 #megameters squared

# Function to preform SG estimation
region_pred = function(p, variables, train_data){
  i <<- i+1
  Y = c()
  lam_means = c()
  for(region in unique(train_data$region_id)){
    # Subset the train data to the region and the specific variables
    rgn_data = train_data[train_data$region_id == region, variables]
    
    # Fit the model
    # Create a vector of the predicted lambda values (parameters*vars)
    # for each fire in the region
    lam = rowSums(p*rgn_data) #how many fires per year per square mile
    #cat("lambda is ", lam[1], "\n") #look to see how big the lambdas are
    
    # Evaluate results
    #TODO: if any lambda is negative, return a HUGE number
    y = (sum(1/lam) - s*t)^2
    # cat("sum is", sum(1/lam), "\n")
    Y = c(Y, y) # sum(ns/lam-T)^2
    lam_means = c(lam_means, mean(lam)) #mean(lam)-the average likelihood of fires that occurred given their conditions
  }
  vals[[i]] <<- lam_means
  #cat("SG stat is ", sum(Y), "\n") #this has to be going down bc optim
  sum(Y)
} 

# Function to prepare and fit SG model with specified hyperparameters and variables
fit_sg = function(train_data, vars, p = .00001, n_plots = 5, opt_p = TRUE){
  if ((i != 0) | (length(vals) != 0)){
    print("TRY AGAIN: reset i and vals for plots")
    return(0)
  }
  
  # Select the optimal starting p values (unless opt_p is False)
  p1 = rep.int(p, length(vars))
  if (opt_p==TRUE){
    opt_p = find_optimal_p_order(train_data, vars)
    p1 = rep.int(opt_p, length(vars))
    cat("Optimal p is ", opt_p, "\n")
  }
  
  # Fit the model
  if (length(vars) == 1){
    model_results = optim(p, region_pred, method = 'Brent', lower = 0, upper = opt_p*10,
                          variables = vars, train_data = train_data)
  } else {
    model_results = optim(p1, region_pred, variables = vars, train_data = train_data)
  }
  
  # Analyze model fitting
  if (model_results$convergence == 1){
    print("Warning: model did not converge")
  }
  cat('Final SG value:', model_results$value, "\n")
  # Print graphs of region estimates for n iterations
  idx = as.integer(seq(1, length(vals), length.out = n_plots))
  for (x in idx){
    values_df = as.data.frame(vals[[x]])
    colnames(values_df)[1] = "fires"
    values_df$region_id = unique(train_data$region_id)
    create_heatmap(viz_df, values_df, paste("Lambda Region Est. Iter.", x, "of", length(vals)))
  }
  
  model_results
}

# Function to determine the p value that produces the smallest SG value
find_optimal_p_order = function(train_data, vars){
  init_p_orders = c(100000, 10000, 1000, 100, 10, 1, .1)
  sg_stats = c()
  for (p in init_p_orders){
    p1 = rep.int(p, length(vars))
    if (length(vars) == 1){
      model_results = optim(p, region_pred, method = 'Brent', lower = 0, upper = p*10,
                            variables = vars, train_data = train_data)
    } else {
      model_results = optim(p1, region_pred, variables = vars, train_data = train_data)
    }
    sg_stats = c(sg_stats, model_results$value)
  }
  opt_p = init_p_orders[which(sg_stats == min(sg_stats))]
  if (length(opt_p) == 1){
    opt_p
  }
  else{
    opt_p[length(opt_p)]
  }
}

# Function to evaluate fitted model on test dataset
test_model = function(train_data, test_data, vars, fitted_model, firetype){
  test_data <- subset(test_data, region_id %in% unique(train_data$region_id))
  # Heatmap of true values
  values_df = as.data.frame(table(train_data$region_id, dnn = list("region_id"))/(s*t), responseName = "fires")
  create_heatmap(viz_df, values_df, paste0(paste(vars, collapse  = '_'),"_", firetype, "_true"),
                 clrs = c("grey87", "darkolivegreen"))
  
  # Heatmap of predicted values
  test_data$rate_estimate = as.matrix(test_data[,vars]) %*% fitted_model$par 
  pred_values_df = aggregate(test_data$rate_estimate, list(test_data$region_id), mean)
  colnames(pred_values_df) <- c("region_id", "fires")
  create_heatmap(viz_df, pred_values_df, paste0(paste(vars, collapse  = '_'),"_", firetype, "_predicted"))
  
  # Density plot of residuals
  colnames(pred_values_df) <- c("region_id", "predictions")
  values_df = merge(values_df, pred_values_df, by = 'region_id', all.x = TRUE)
  values_df$fires = values_df$predictions - values_df$fires
  
  png(file=paste0("~/MS Stats/Research/Wildfire/results/", t, '_density.png'), width=800, height=600)
  plot(density(values_df$fires), main = '', xlab='residual', ylab = 'density')
  dev.off()
  
  # Heatmap of residual values
  values_df$fires = abs(values_df$fires)
  mean_res = mean(abs(values_df$fires))
  t = paste0(paste(vars, collapse  = '_'),"_", firetype, "_residual")
  create_heatmap(viz_df, values_df, t,
                 clrs = c("grey82", "grey40"))
  cat('Mean Absolute Residual Value', mean_res)
  
  values_df$fires
}



### MODEL IMPLEMENTATION

# Fitting all of California fires to one model

all_vars = c('constant', 'AWND', 'TMAX', 'EVAP', 'TMIN', 'PRCP_ROLLING')

i <- 0
vals <- list()
model0 = fit_sg(data, all_vars)
res0 = test_model(data, test_data, all_vars, model0, "all")

i <- 0
vals <- list()
natural_model0 = fit_sg(natural_df, all_vars) #fitting only natural fires
resnat0 = test_model(natural_df, test_data, all_vars, natural_model0, "natural")

i <- 0
vals <- list()
unnatural_model0 = fit_sg(unnatural_df, all_vars) #fitting only human-caused fires
resunnat0 = test_model(unnatural_df, test_data, all_vars, unnatural_model0, "unnatural")

model0$par 
natural_model0$par
unnatural_model0$par

model0$par *sapply(data[, all_vars], quantile, prob=.9, names=FALSE)
natural_model0$par *sapply(natural_df[, all_vars], quantile, prob=.9, names=FALSE)
unnatural_model0$par *sapply(unnatural_df[, all_vars], quantile, prob=.9, names=FALSE)



# Repeating for SoCal fires only
socal_regions = sort(unique(data$region_id))[1:19]
socal_data = data[data$region_id %in% socal_regions, ]
socal_natural_df = socal_data[socal_data$LGHTNG == 1,]
socal_unnatural_df = socal_data[socal_data$LGHTNG == 0,]
socal_test_data = test_data[test_data$region_id %in% socal_regions,]

values_df = as.data.frame(table(data$region_id, dnn = list("region_id")), responseName = "fires")
socal_values_df = as.data.frame(table(socal_data$region_id, dnn = list("region_id")), responseName = "fires")
summary(values_df$fires)
summary(socal_values_df$fires)

model_vars = c('constant')

i <- 0  #i and vals are global variables
vals <- list() #used for plotting model fitting progress
model = fit_sg(socal_data, model_vars)
test_model(socal_data, socal_test_data, model_vars, model, "all")

i <- 0  #i and vals are global variables
vals <- list() #used for plotting model fitting progress
natural_model = fit_sg(socal_natural_df, model_vars)
test_model(socal_natural_df, socal_test_data, model_vars, natural_model)

i <- 0
vals <- list()
unnatural_model = fit_sg(socal_unnatural_df, model_vars)
test_model(socal_unnatural_df, socal_test_data, model_vars, unnatural_model)

model$par
natural_model$par
unnatural_model$par

model$par *sapply(data[, model_vars], quantile, prob=.9, names=FALSE)
natural_model$par *sapply(natural_df[, model_vars], quantile, prob=.9, names=FALSE)
unnatural_model$par *sapply(unnatural_df[, model_vars], quantile, prob=.9, names=FALSE)











### Chicken-scratch & quick analysis of specific data points


# Plot a sample of the estimates from the two models
data$natural_pred = as.matrix(data[,model_vars]) %*% natural_model$par
data$unnatural_pred = as.matrix(data[,model_vars]) %*% unnatural_model$par
smpl = data[sample(nrow(data), 100), ]
ggplot() + 
  geom_point(data = smpl, aes(x = POO_LONGITUDE, y = POO_LATITUDE, 
                              size = natural_pred)) 
ggplot() + 
  geom_point(data = smpl, aes(x = POO_LONGITUDE, y = POO_LATITUDE, 
                              size = unnatural_pred)) 

model$par
natural_model$par
unnatural_model$par



# Case 1: variables = medians
sample.case = apply(data[,variables],2,median)
sum(natural_model$par * sample.case)
sum(unnatural_model$par * sample.case)
# Natural fire less likely than unnatural fire

# Case 2: variables = medians, but HWY_DSTNC further
sample.case["HWY_DSTNC"] = 10
sum(natural_model$par * sample.case)
sum(unnatural_model$par * sample.case)
# Compared to case 1, natural fire now more likely, unnatural less likely

# Case 3: a campfire that burned 75000 acres
sample.case3 = data[data$OBJECTID == 134429, variables]
sum(natural_model$par * sample.case3)
sum(unnatural_model$par * sample.case3)

# Examining to see if there are any instances where the 
# likelihood of a natural fire is greater than a unnatural fire
samples = data[data$natural_predict > data$unnatural_predict, ]
dim(samples)