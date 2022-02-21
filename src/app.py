from h2o_wave import Q, ui, app, main, data
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import pandas as pd

def deletePages(q):
    pages = ['homepage', 'loaddata','error','showdata', 'clusterdata', 'supportapp', 'aboutapp', 'plot', 'plot1']
    for page in pages:
        del q.page[page]

async def displayError(q, message):
    q.page['error'] = ui.preview_card(
        name='preview_card',
        box='3 4 8 1',
        image='https://images.pexels.com/photos/960137/pexels-photo-960137.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940',
        title=message,
    )
    await q.page.save()

async def uploadSuccess(q, message):
    q.page['error'] = ui.preview_card(
        name='preview_card',
        box='3 4 8 1',
        image='https://cdn.pixabay.com/photo/2013/08/28/11/47/leaf-176722__340.jpg',
        title=message,
    )
    await q.page.save()

def make_markdown_table(fields, rows):
    return '\n'.join([
        make_markdown_row(fields),
        make_markdown_row('---' * len(fields)),
        '\n'.join([make_markdown_row(row) for row in rows]),
    ])
def make_markdown_row(values):
    return f"| {' | '.join([str(x) for x in values])} |"

async def showData(q,df):
    q.page['showdata'] = ui.form_card(
        box='3 2 8 6',
        title='Data Set',
        items = [ui.text(make_markdown_table(fields=df.columns.tolist(),rows=df.values.tolist()))],
    )
    await q.page.save()

async def homePage(q):
    q.page["homepage"] = ui.form_card(
        box='3 2 8 6',
        items=[
            ui.text_l("K-means Clustering..."),
            ui.text('''
* K-means clustering is one of the simplest and popular unsupervised machine learning algorithms.

* Typically, unsupervised algorithms make inferences from datasets using only input vectors without referring to known, or labelled, outcomes.

* A cluster refers to a collection of data points aggregated together because of certain similarities.

* You’ll define a target number k, which refers to the number of centroids you need in the dataset. A centroid is the imaginary or real location representing the center of the cluster.

* Every data point is allocated to each of the clusters through reducing the in-cluster sum of squares.

* In other words, the K-means algorithm identifies k number of centroids, and then allocates every data point to the nearest cluster, while keeping the centroids as small as possible.

* The ‘means’ in the K-means refers to averaging of the data; that is, finding the centroid.
            '''),
            ui.text_l("How the K-means algorithm works..."),
            ui.text('''
* To process the learning data, the K-means algorithm in data mining starts with a first group of randomly selected centroids, which are used as the beginning points for every cluster, and then performs iterative (repetitive) calculations to optimize the positions of the centroids.

* It halts creating and optimizing clusters when either:

    1. The centroids have stabilized — there is no change in their values because the clustering has been successful.
    2. The defined number of iterations has been achieved.
            '''),
        ],
    )
    await q.page.save()

async def loadData(q):
    q.page['loaddata'] = ui.form_card(
        box='3 2 8 6',
        items=[
            ui.text_xl('Upload Data'),
            ui.file_upload(name='data_file', label='Upload', multiple=False),
        ]
    )
    if (q.args.data_file):
        global colList, localpath
        if (q.args.data_file[0][-3:] == "csv"):
            localpath = await q.site.download(q.args.data_file[0], '../data.csv')
            data = pd.read_csv(localpath)

            numerical_columns = [num_col for num_col in data.columns if data[num_col].dtype in ['int','float','int64','float64']]
            if (len(numerical_columns) == 0):
                deletePages(q)
                await displayError(q, "Error! No Numerical Features Available in the Dataset")
            else:
                numerical_df = data[numerical_columns].dropna()
                filepath = Path('../numerical_data.csv')  
                numerical_df.to_csv(filepath)

                deletePages(q)
                await uploadSuccess(q, "Data uploaded Successfully.")
        else:
            deletePages(q)
            await displayError(q, "Error! Upload a CSV file")

    await q.page.save()

async def clusterData(q,columns, df):
    choices = []
    kVal = []
    for col in columns:
        choices.append(ui.choice(col, col))
    for k in range(1,11):
        kVal.append(ui.choice(str(k), str(k)))
    print(len(choices))
    q.page['clusterdata'] = ui.form_card(
        box='3 2 8 6',
        items=[
            ui.text_xl('Select Cluster Parameters'),
            ui.dropdown(name='Kdropdown', label='K - Value', value=str(1), required=True, choices=kVal),
            ui.dropdown(name='Xdropdown', label='X - Axis', value=columns[0], required=True, choices=choices),
            ui.dropdown(name='Ydropdown', label='Y - Axis', value=columns[0], required=True, choices=choices),
            ui.button(name='cluster', label='Run Clusters',primary=True),
        ]
    )
    if (q.args.cluster):
        deletePages(q)
        await plotGraph(q, q.args.Kdropdown, q.args.Xdropdown, q.args.Ydropdown, df, columns)
    await q.page.save()

async def plotGraph(q,k,x,y, d_f, c):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(d_f)
    data_scaled = pd.DataFrame(scaled, columns=d_f.columns)
    kmeans = KMeans(n_clusters=int(k), random_state=111)
    kmeans.fit(data_scaled)
    cluster_labels = kmeans.labels_
    df1 = d_f
    df1[k +'_clusters'] = cluster_labels
    print(df1.head())

    df_point =  df1.loc[:200,[x,y,k+'_clusters']]
    print(df_point.head())
    q.page['plot'] = ui.plot_card(
        box='3 2 8 5',
        title=k + 'Clusters',
        data=data('x y k_clusters', 200, rows=df_point.values.tolist()),
        plot=ui.plot([
            ui.mark(type='point', x='=x', y='=y', color='=k_clusters', x_title=x, y_title=y)
        ])
    )
    print("Plot Done")
    q.page['plot1'] = ui.form_card(
        box='3 7 8 1',
        items=[
            ui.button(name='backBtn', label='Back',primary=False),
        ]
    )
    print("Back Done")
    if (q.args.backBtn):
        deletePages(q)
        await clusterData(q, c, d_f)
    await q.page.save()

async def supportApp(q):

    persona_pic = 'https://media-exp1.licdn.com/dms/image/C5103AQHIxPRSjP7NqA/profile-displayphoto-shrink_800_800/0/1517573086852?e=1649894400&v=beta&t=EmeQgPZJDEO8hn_Tq4Ta-iZBvbdEYkDlqiakwVbbsR8'
    q.page['supportapp'] = ui.wide_article_preview_card(
        box='3 2 8 6',
        persona=ui.persona(title='Shamil Prematunga', subtitle='Software Engineer',
                            image=persona_pic, caption='caption'),
        image='https://images.pexels.com/photos/1269968/pexels-photo-1269968.jpeg?auto=compress',
        title='Shamil Prematunga',
        caption='''
    I am a Software Engineer who is willing to study Machine Learning and Artificial Intelligence. As a practice, 
    I am trying to create my own AI applications to demonstrate some basic concepts of ML. I am using H2O Wave as 
    a tool to build up applications easily. If anyone is interested to get support regarding this application feel 
    free to leave a message via LinkedIn.
        ''',
        items=[
            ui.inline(justify='end', items=[
                ui.link(label='Shamil Prematunga (LinkedIn)', path='https://www.linkedin.com/in/shamil-prematunga-139b51158/', target='_blank'),
            ])
        ]
    )

    await q.page.save()

async def aboutApp(q):

    persona_pic = 'https://media.istockphoto.com/vectors/letter-k-with-heart-icon-vector-id665103306?k=20&m=665103306&s=612x612&w=0&h=uyuUsKo6MEE6p2dG23eXuGaco2YgmT53KY5rDApSi1Q='
    q.page['aboutapp'] = ui.wide_article_preview_card(
        box='3 2 8 6',
        persona=ui.persona(title='K-Means App', subtitle='Make it easy',
                            image=persona_pic, caption='caption'),
        image='https://images.pexels.com/photos/669615/pexels-photo-669615.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500',
        title='About Application',
        caption='''
    This application is mainly focusing on unsupervised learning applications. Out of the main four areas where unsupervised 
    learning is currently applied, this application uses K-mean clustering. Here we can upload our dataset and check the 
    distribution of the data points related to the two parameters which we can select by ourselves. There is a capability to 
    select “K-value” in the algorithm and finally check how those points are being clustered. 

    This application uses H2O Wave 0.20.0 release as the development platform. 

        ''',
        items=[
            ui.inline(justify='end', items=[
                ui.links(label='Reference Links', width='200px', items=[
                    ui.link(label='Unsupervised Learning', path='https://h2o.ai/blog/an-introduction-to-unsupervised-machine-learning/', target='_blank'),
                    ui.link(label='K-Means Algorithm', path='https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1', target='_blank'),
                    ui.link(label='H2O Wave', path='https://lnkd.in/guyWjeGV', target='_blank'),
                ]),
            ])
        ]
    )

    await q.page.save()

@app('/kMeans')
async def serve(q:Q):
    hash = q.args['#']
    q.page['meta'] = ui.meta_card(
        box='',
        themes=[
            ui.theme(
                name='my-awesome-theme',
                primary='#FF9F1C',
                text='#e8e1e1',
                card='#000000',
                page='#000000',
            )
        ],
        theme='my-awesome-theme'
    )
    q.page['head'] = ui.header_card(
        box='3 1 8 1',
        title='K-Mean Clustering',
        subtitle='Check the clusters in your own dataset.',
    )
    q.page['nav'] = ui.nav_card(
        box='1 1 2 8',
        title='K-Means App',
        subtitle='Make it easy',
        image='https://media.istockphoto.com/vectors/letter-k-with-heart-icon-vector-id665103306?k=20&m=665103306&s=612x612&w=0&h=uyuUsKo6MEE6p2dG23eXuGaco2YgmT53KY5rDApSi1Q=',
        items=[
            ui.nav_group('Menu', items=[
                ui.nav_item(name='#home', label='Home'),
                ui.nav_item(name='#load', label='Load Data'),
                ui.nav_item(name='#show', label='Show Data'),
                ui.nav_item(name='#cluster', label='Clusters'),
            ]),
            ui.nav_group('Help', items=[
                ui.nav_item(name='#about', label='About', icon='Info'),
                ui.nav_item(name='#support', label='Support', icon='Help'),
            ])
        ],
    )
    q.page['footer'] = ui.footer_card(box='3 8 8 1', caption='Made with 💛 by Shamil Prematunga.')
    if ((hash == 'home') or (hash == 'load') or (hash == 'show') or (hash == 'cluster') or (hash == 'about') or (hash == 'support')):
        if (hash == 'home'):
            deletePages(q)
            await homePage(q)
        elif (hash == 'load'):
            deletePages(q)
            await loadData(q)
        elif (hash == 'show'):
            try:
                data = pd.read_csv('../numerical_data.csv',index_col=0)
                df = data.loc[:200,:]
            except:
                deletePages(q)
                await displayError(q, "Error! No data available to display.")
            else:
                deletePages(q)
                await showData(q,df)
        elif (hash == 'cluster'):
            try:
                data = pd.read_csv('../numerical_data.csv',index_col=0)
                df = data.loc[:200,:]
                numerical_columns = [num_col for num_col in df.columns if df[num_col].dtype in ['int','float','int64','float64']]
            except:
                deletePages(q)
                await displayError(q, "Error! No clusters available to display.")
            else:
                deletePages(q)
                await clusterData(q,numerical_columns,df)
        elif (hash == 'about'):
            deletePages(q)
            await aboutApp(q)
        elif (hash == 'support'):
            deletePages(q)
            await supportApp(q)
    else:
        deletePages(q)
        await homePage(q)
    await q.page.save()
